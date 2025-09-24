import os
from pathlib import Path
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset

from src.loftr import LoFTR
from src.config.default import get_cfg_defaults
from src.utils.misc import lower_config, flattenList
from src.datasets.megadepth import MegaDepthDataset
from src.utils.metrics import compute_symmetrical_epipolar_errors, compute_pose_errors, aggregate_metrics
from src.utils.comm import gather


# ------------------- Metrics -------------------
def compute_metrics(batch, config):
    compute_symmetrical_epipolar_errors(batch)
    compute_pose_errors(batch, config)
    rel_pair_names = list(zip(*batch['pair_names']))
    bs = batch['image0'].size(0)
    metrics = {
        'identifiers': ['#'.join(rel_pair_names[b]) for b in range(bs)],
        'epi_errs': [(batch['epi_errs'].reshape(-1, 1))[batch['m_bids'] == b].reshape(-1).cpu().numpy() for b in
                     range(bs)],
        'R_errs': batch['R_errs'],
        't_errs': batch['t_errs'],
        'inliers': batch['inliers'],
        'num_matches': [batch['mconf'].shape[0]],
    }
    return {'metrics': metrics}, rel_pair_names


# ------------------- 主评估函数 -------------------
def evaluate_model(data_dir, model, cfg):
    scene_list = "/home/wby/project/work/EfficientLoFTR/assets/megadepth_test_1500_scene_info/"
    scene_list_path = os.path.join(scene_list, 'megadepth_test_1500.txt')
    with open(scene_list_path, 'r') as f:
        npz_paths = [os.path.join(scene_list, f"{name.strip()}.npz") for name in f.readlines()]

    dataset = ConcatDataset([MegaDepthDataset(
        root_dir=data_dir,
        npz_path=npz_path,
        min_overlap_score=0,
        mode='test',
        img_resize=cfg.LOFTR.COARSE.NPE[2],
        df=8,
        img_padding=True,
        depth_padding=True,
        augment_fn=None,
        coarse_scale=1 / cfg.LOFTR.RESOLUTION[0],
        fp16=cfg.DATASET.FP16
    ) for npz_path in npz_paths])

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)
    device = next(model.parameters()).device

    inference_times = []
    outputs = []
    warmup = False

    with torch.no_grad():
        for i, data in enumerate(tqdm(dataloader, desc="Evaluating", unit="batch")):
            # 只保留模型输入和 metrics 相关字段
            data = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in data.items()}

            # warmup
            if not warmup:
                with torch.autocast(enabled=True, device_type='cuda'):
                    for _ in range(10):
                        model(data)
                warmup = True

            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.autocast(enabled=True, device_type='cuda'):
                start_event.record()
                model(data)
                end_event.record()
            torch.cuda.synchronize()
            inference_times.append(start_event.elapsed_time(end_event))

            # 只计算 metrics
            ret_dict, _ = compute_metrics(data, cfg)
            outputs.append(ret_dict)

    # 汇总 metrics
    _metrics = [o['metrics'] for o in outputs]
    metrics = {k: flattenList(gather(flattenList([_me[k] for _me in _metrics]))) for k in _metrics[0]}
    val_metrics_4tb = aggregate_metrics(metrics, cfg.TRAINER.EPI_ERR_THR, config=cfg)
    print(val_metrics_4tb)

    return {"Average Inference Time (ms)": float(torch.tensor(inference_times).mean().item())}


# ------------------- 主程序 -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='data/megadepth/test',
                        help="Megadepth test data root dir")
    parser.add_argument("--main_cfg_path", type=str, default="configs/loftr/eloftr_full.py")
    parser.add_argument("--weights", type=str, default='weights/mloftr.ckpt')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.main_cfg_path)
    cfg.LOFTR.COARSE.NPE = [832, 832, 1184, 1184]
    cfg.LOFTR.MATCH_COARSE.THR = 0.2

    _config = lower_config(cfg)
    model = LoFTR(config=_config['loftr'])
    if args.weights:
        state_dict = torch.load(args.weights, map_location='cpu')['state_dict']
        model.load_state_dict(state_dict, strict=False)
    model = model.cuda().eval()

    results = evaluate_model(args.data_dir, model, cfg)
    print("Evaluation Results:", results)