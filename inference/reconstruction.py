import os
import copy
import tempfile
import glob
import datetime
import pickle
import json
import numpy as np
import torch

from utils.raw_data_processing import load_npy_images
from mast3r.cloud_opt.sparse_ga import plot_2d_matrix_v2, sparse_global_alignment
from mast3r.model import AsymmetricMASt3R
from mast3r.demo import (
    SparseGAState,
    print_reconstruction_params,
    get_3D_model_from_scene,
)
from mast3r.image_pairs import make_pairs
from model.lora_model import LoRAMast3r
from utils.save_transforms_json import save_json_file
from utils.visualize_pointclouds_html import generate_html, start_http_server
from dust3r.utils.image import load_images


def load_model(args):
    """Load model for inference."""
    if args.pairs_pkl_path is not None and os.path.exists(args.pairs_pkl_path):
        return None
    elif args.finetuned_model is None:
        model = AsymmetricMASt3R.from_pretrained("naver/" + args.model).to(args.device)
    elif args.finetuned_model.endswith("_lora.pth"):
        model = LoRAMast3r.from_pretrained("naver/" + args.model)
        model.load_pretrained_lora(args.finetuned_model)
        model = model.to(args.device)
    elif args.finetuned_model.endswith("_mast3r.pth"):
        model = AsymmetricMASt3R.from_pretrained("naver/" + args.model).to(args.device)
        model.load_state_dict(torch.load(args.finetuned_model))
    else:
        raise ValueError("finetuned_model should end with _lora.pth or _mast3r.pth")
    return model


def prepare_images(args):
    """Load and prepare images for reconstruction."""
    if args.filelist[0].endswith(".npy"):
        imgs = load_npy_images(args.filelist, max_value=16383)
        filelist = args.filelist
    else:
        imgs = load_images(args.filelist, size=args.image_size, verbose=not args.silent)
        filelist = args.filelist


    print(f"Done loading images. Loaded {len(imgs)} images.")

    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]["idx"] = 1
        filelist = [filelist[0], filelist[0] + "_2"]

    return imgs, filelist


def create_image_pairs(args, imgs, filelist, model):
    """Create image pairs for reconstruction."""
    scene_graph_params = [args.scenegraph_type]
    if args.scenegraph_type in ["swin", "logwin"]:
        scene_graph_params.append(str(args.winsize))
    elif args.scenegraph_type == "oneref":
        scene_graph_params.append(str(args.refid))
    elif args.scenegraph_type == "retrieval":
        scene_graph_params.append(str(args.winsize))
        scene_graph_params.append(str(args.refid))
    elif args.scenegraph_type == "swinret":
        scene_graph_params += [str(args.winsize), str(getattr(args, "ret_na", 20)), str(getattr(args, "ret_k", 10))]

    if args.scenegraph_type in ["swin", "logwin", "swinret"] and not args.win_cyclic:
        scene_graph_params.append("noncyclic")
    scene_graph = "-".join(scene_graph_params)

    if args.pairs_pkl_path is not None:
        pairs = None
    else:
        if args.sim_matrix is not None and os.path.exists(args.sim_matrix):
            sim_matrix = np.load(args.sim_matrix)
            print(f"loaded sim matrix from {args.sim_matrix}")
        elif "retrieval" in args.scenegraph_type or args.scenegraph_type == "swinret":
            assert args.retrieval_model is not None
            from mast3r.retrieval.processor import Retriever
            retriever = Retriever(
                args.retrieval_model, backbone=model, device=args.device
            )
            print("Done creating retrieval network.")
            with torch.no_grad():
                sim_matrix = retriever(filelist)
            print(f"done creating sim matrix. Shape of sim_matrix: {sim_matrix.shape}")
            np.save(os.path.join(args.outdir, "sim_matrix.npy"), sim_matrix)
            del retriever
            torch.cuda.empty_cache()
            print("done with torch.cuda.empty_cache()")
            plot_2d_matrix_v2(
                sim_matrix,
                "Similarity Matrix",
                os.path.join(args.outdir, "sim_matrix.png"),
                normalize=False,
                log_scale=True,
                vmin=0,
                vmax=2e-1,
            )
            print("Done plotting sim_matrix. Shape: ", sim_matrix.shape)
        else:
            sim_matrix = None
        pairs = make_pairs(
            imgs,
            scene_graph=scene_graph,
            prefilter=None,
            symmetrize=True,
            sim_mat=sim_matrix,
        )

    return pairs


def run_reconstruction(args, filelist, pairs, model):
    """Run sparse global alignment for 3D reconstruction."""
    if args.optim_level == "coarse":
        args.niter2 = 0

    if args.optimizer_dtype == "float16":
        dtype = torch.float16
    elif args.optimizer_dtype == "float32":
        dtype = torch.float32

    scene = sparse_global_alignment(
        filelist,
        pairs,
        args.cache_dir,
        model,
        lr1=args.lr1,
        niter1=args.niter1,
        lr2=args.lr2,
        niter2=args.niter2,
        device=args.device,
        opt_depth="depth" in args.optim_level,
        shared_intrinsics=True,
        matching_conf_thr=args.matching_conf_thr,
        init_focal=args.init_focal,
        focal_reg_w=args.focal_reg_w,
        plot_matches=args.plot_matches,
        pairs_pkl_path=args.pairs_pkl_path,
        dtype=dtype,
        scale_norm=getattr(args, "ga_scale_norm", "min"),
        **({"init": args.init_poses} if getattr(args, "init_poses", None) else {}),
    )

    return scene


def export_scene(args, scene, elapsed_s):
    """Export scene to various formats."""
    if (
        args.current_scene_state is not None
        and not args.current_scene_state.should_delete
        and args.current_scene_state.outfile_name is not None
    ):
        outfile_name = args.current_scene_state.outfile_name
    else:
        outfile_name = tempfile.mktemp(suffix="_scene.glb", dir=args.outdir)

    print("Getting scene state from SparseGAState()")
    scene_state = SparseGAState(
        scene, args.gradio_delete_cache, args.cache_dir, outfile_name
    )

    save_json_file(scene_state, os.path.join(args.outdir, "transforms.json"), elapsed_s)

    try:
        outfile = get_3D_model_from_scene(
            args.silent,
            scene_state,
            args.min_conf_thr,
            args.as_pointcloud,
            args.mask_sky,
            args.clean_depth,
            args.transparent_cams,
            args.cam_size,
            args.TSDF_thresh,
        )

        html_file_path = os.path.join(args.outdir, "glb_viewer.html")
        generate_html(outfile, html_file_path)

        if args.start_server:
            start_http_server(args.outdir)
    except Exception as e:
        print(f"Error in get_3D_model_from_scene: {e}")


def reconstruct_scene(args):
    """Main reconstruction pipeline."""
    now = datetime.datetime.now()
    args.outdir = os.path.join(args.outdir, now.strftime("%m-%d-%H-%M-%S"))
    os.makedirs(args.outdir, exist_ok=True)

    args_dict = vars(args)
    with open(os.path.join(args.outdir, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

    # Process filelist
    args.filelist = sorted(glob.glob(args.filelist))

    if args.subset_list is not None:
        with open(args.subset_list, "r") as f:
            subset_list = f.readlines()
        subset_list = [x.strip() for x in subset_list]
        args.filelist = [x for x in args.filelist if os.path.basename(x) in subset_list]
    else:
        if args.images_start >= 0 and args.images_end > 0:
            args.filelist = args.filelist[args.images_start : args.images_end]
        if args.images_end > 0 and len(args.filelist) > args.images_end:
            args.filelist[: args.images_end]
        if args.subsample > 1:
            args.filelist = args.filelist[args.subsample // 2 :: args.subsample]

    print(f"subsampled filelist len: {len(args.filelist)}")

    # Load model
    model = load_model(args)

    # Setup cache directory
    if (
        args.current_scene_state is not None
        and not args.current_scene_state.should_delete
        and args.current_scene_state.cache_dir is not None
    ):
        cache_dir = args.current_scene_state.cache_dir
    elif args.gradio_delete_cache:
        cache_dir = tempfile.mkdtemp(suffix="_cache", dir=args.outdir)
    else:
        cache_dir = os.path.join(args.outdir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    args.cache_dir = cache_dir

    # Load images
    imgs, filelist = prepare_images(args)

    print_reconstruction_params(args)

    # Create image pairs
    pairs = create_image_pairs(args, imgs, filelist, model)

    # Run reconstruction
    scene = run_reconstruction(args, filelist, pairs, model)

    # Save scene
    with open(os.path.join(args.outdir, "scene.pkl"), "wb") as f:
        pickle.dump(scene, f)

    # Export scene
    elapsed_time = datetime.datetime.now() - now
    elapsed_s = int(elapsed_time.total_seconds())
    export_scene(args, scene, elapsed_s)
