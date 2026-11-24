#!/usr/bin/env python3
import torch
import json
import os
import sys
import argparse
import yaml

import loralib as lora  # Ensure you have LoRA installed

# Note: mast3r.model is an external dependency
from mast3r.model import AsymmetricMASt3R


class LoRAMast3r(AsymmetricMASt3R):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.args = None

    def setup_lora_model(self, args):
        self.args = args

        self.replace_encoder_attention_with_lora()
        self.replace_decoder_with_lora()
        self.replace_head_local_feature_mlp_with_lora()
        self.replace_head_dpt_with_lora()
        lora.mark_only_lora_as_trainable(self)

    def freeze_encoder_attn(self):
        for idx, block in enumerate(self.enc_blocks):
            for param in block.parameters():
                param.requires_grad = False

    def freeze_lora_layers(self):
        for idx, block in enumerate(self.enc_blocks):
            for param in block.parameters():
                param.requires_grad = False
        for idx, block in enumerate(self.dec_blocks):
            for param in block.parameters():
                param.requires_grad = False

    def replace_encoder_attention_with_lora(self):
        # legacy
        if hasattr(self.args, "lora_rank_encoder_attn"):
            setattr(
                self.args,
                "lora_rank_encoder_attn_qkv",
                self.args.lora_rank_encoder_attn,
            )
            setattr(
                self.args,
                "lora_alpha_encoder_attn_qkv",
                self.args.lora_alpha_encoder_attn,
            )

        for ind, block in enumerate(self.enc_blocks):
            if (
                hasattr(self.args, "lora_rank_encoder_attn_qkv")
                and self.args.lora_rank_encoder_attn_qkv > 0
            ):
                lora_attn_qkv = lora.Linear(
                    block.attn.qkv.in_features,
                    block.attn.qkv.out_features,
                    r=self.args.lora_rank_encoder_attn_qkv,
                    lora_alpha=self.args.lora_alpha_encoder_attn_qkv,
                )
                lora_attn_qkv.weight.data.copy_(block.attn.qkv.weight.data)
                if block.attn.qkv.bias is not None:
                    lora_attn_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                setattr(self.enc_blocks[ind].attn, "qkv", lora_attn_qkv)

            if (
                hasattr(self.args, "lora_rank_encoder_attn_proj")
                and self.args.lora_rank_encoder_attn_proj > 0
            ):
                lora_attn_proj = lora.Linear(
                    block.attn.proj.in_features,
                    block.attn.proj.out_features,
                    r=self.args.lora_rank_encoder_attn_proj,
                    lora_alpha=self.args.lora_alpha_encoder_attn_proj,
                )
                lora_attn_proj.weight.data.copy_(block.attn.proj.weight.data)
                if block.attn.proj.bias is not None:
                    lora_attn_proj.bias.data.copy_(block.attn.proj.bias.data)

                setattr(self.enc_blocks[ind].attn, "proj", lora_attn_proj)

            if (
                hasattr(self.args, "lora_rank_encoder_mlp_fc1")
                and self.args.lora_rank_encoder_mlp_fc1 > 0
            ):
                lora_mlp_fc1 = lora.Linear(
                    block.mlp.fc1.in_features,
                    block.mlp.fc1.out_features,
                    r=self.args.lora_rank_encoder_mlp_fc1,
                    lora_alpha=self.args.lora_alpha_encoder_mlp_fc1,
                )
                lora_mlp_fc1.weight.data.copy_(block.mlp.fc1.weight.data)
                if block.mlp.fc1.bias is not None:
                    lora_mlp_fc1.bias.data.copy_(block.mlp.fc1.bias.data)

                setattr(self.enc_blocks[ind].mlp, "fc1", lora_mlp_fc1)

            if (
                hasattr(self.args, "lora_rank_encoder_mlp_fc2")
                and self.args.lora_rank_encoder_mlp_fc2 > 0
            ):
                lora_mlp_fc2 = lora.Linear(
                    block.mlp.fc2.in_features,
                    block.mlp.fc2.out_features,
                    r=self.args.lora_rank_encoder_mlp_fc2,
                    lora_alpha=self.args.lora_alpha_encoder_mlp_fc2,
                )
                lora_mlp_fc2.weight.data.copy_(block.mlp.fc2.weight.data)
                if block.mlp.fc2.bias is not None:
                    lora_mlp_fc2.bias.data.copy_(block.mlp.fc2.bias.data)

                setattr(self.enc_blocks[ind].mlp, "fc2", lora_mlp_fc2)

    def replace_head_local_feature_mlp_with_lora(self):
        if hasattr(self.args, "lora_rank_local_feature_mlp"):  # legacy
            setattr(
                self.args,
                "lora_rank_head_mlp_fc1",
                self.args.lora_rank_local_feature_mlp,
            )
            setattr(
                self.args,
                "lora_rank_head_mlp_fc2",
                self.args.lora_rank_local_feature_mlp,
            )
            setattr(
                self.args,
                "lora_alpha_head_mlp_fc1",
                self.args.lora_alpha_local_feature_mlp,
            )
            setattr(
                self.args,
                "lora_alpha_head_mlp_fc2",
                self.args.lora_alpha_local_feature_mlp,
            )

        for downstream_head in (self.downstream_head1, self.downstream_head2):
            if (
                hasattr(self.args, "lora_rank_head_mlp_fc1")
                and self.args.lora_rank_head_mlp_fc1 > 0
            ):
                lora_fc1 = lora.Linear(
                    in_features=downstream_head.head_local_features.fc1.in_features,
                    out_features=downstream_head.head_local_features.fc1.out_features,
                    r=self.args.lora_rank_head_mlp_fc1,
                    lora_alpha=self.args.lora_alpha_head_mlp_fc1,
                )

                lora_fc1.weight.data.copy_(
                    downstream_head.head_local_features.fc1.weight.data
                )
                if downstream_head.head_local_features.fc1.bias is not None:
                    lora_fc1.bias.data.copy_(
                        downstream_head.head_local_features.fc1.bias.data
                    )

                downstream_head.head_local_features.fc1 = lora_fc1

            if (
                hasattr(self.args, "lora_rank_head_mlp_fc2")
                and self.args.lora_rank_head_mlp_fc2 > 0
            ):
                lora_fc2 = lora.Linear(
                    in_features=downstream_head.head_local_features.fc2.in_features,
                    out_features=downstream_head.head_local_features.fc2.out_features,
                    r=self.args.lora_rank_head_mlp_fc2,
                    lora_alpha=self.args.lora_alpha_head_mlp_fc2,
                )

                lora_fc2.weight.data.copy_(
                    downstream_head.head_local_features.fc2.weight.data
                )
                if downstream_head.head_local_features.fc2.bias is not None:
                    lora_fc2.bias.data.copy_(
                        downstream_head.head_local_features.fc2.bias.data
                    )

                downstream_head.head_local_features.fc2 = lora_fc2

    def replace_decoder_with_lora(self):
        if hasattr(self.args, "lora_rank_decoder_attn"):  # legacy args
            setattr(
                self.args,
                "lora_rank_decoder_attn_qkv",
                self.args.lora_rank_decoder_attn,
            )
            setattr(
                self.args,
                "lora_alpha_decoder_attn_qkv",
                self.args.lora_alpha_decoder_attn,
            )

        dec_blocks_list = [self.dec_blocks, self.dec_blocks2]

        for dec_blocks in dec_blocks_list:
            for ind, block in enumerate(dec_blocks):
                if (
                    hasattr(self.args, "lora_rank_decoder_attn_qkv")
                    and self.args.lora_rank_decoder_attn_qkv > 0
                ):
                    lora_attn_qkv = lora.Linear(
                        block.attn.qkv.in_features,
                        block.attn.qkv.out_features,
                        r=self.args.lora_rank_decoder_attn_qkv,
                        lora_alpha=self.args.lora_alpha_decoder_attn_qkv,
                    )
                    lora_attn_qkv.weight.data.copy_(block.attn.qkv.weight.data)
                    if block.attn.qkv.bias is not None:
                        lora_attn_qkv.bias.data.copy_(block.attn.qkv.bias.data)

                    setattr(dec_blocks[ind].attn, "qkv", lora_attn_qkv)

                if (
                    hasattr(self.args, "lora_rank_decoder_attn_proj")
                    and self.args.lora_rank_decoder_attn_proj > 0
                ):
                    lora_attn_proj = lora.Linear(
                        block.attn.proj.in_features,
                        block.attn.proj.out_features,
                        r=self.args.lora_rank_decoder_attn_proj,
                        lora_alpha=self.args.lora_alpha_decoder_attn_proj,
                    )
                    lora_attn_proj.weight.data.copy_(block.attn.proj.weight.data)
                    if block.attn.proj.bias is not None:
                        lora_attn_proj.bias.data.copy_(block.attn.proj.bias.data)

                    setattr(dec_blocks[ind].attn, "proj", lora_attn_proj)

                if (
                    hasattr(self.args, "lora_rank_decoder_cross_attn_projq")
                    and self.args.lora_rank_decoder_cross_attn_projq > 0
                ):
                    lora_cross_attn_projq = lora.Linear(
                        block.cross_attn.projq.in_features,
                        block.cross_attn.projq.out_features,
                        r=self.args.lora_rank_decoder_cross_attn_projq,
                        lora_alpha=self.args.lora_alpha_decoder_cross_attn_projq,
                    )
                    lora_cross_attn_projq.weight.data.copy_(
                        block.cross_attn.projq.weight.data
                    )
                    if block.cross_attn.projq.bias is not None:
                        lora_cross_attn_projq.bias.data.copy_(
                            block.cross_attn.projq.bias.data
                        )

                    setattr(dec_blocks[ind].cross_attn, "projq", lora_cross_attn_projq)

                if (
                    hasattr(self.args, "lora_rank_decoder_cross_attn_projk")
                    and self.args.lora_rank_decoder_cross_attn_projk > 0
                ):
                    lora_cross_attn_projk = lora.Linear(
                        block.cross_attn.projk.in_features,
                        block.cross_attn.projk.out_features,
                        r=self.args.lora_rank_decoder_cross_attn_projk,
                        lora_alpha=self.args.lora_alpha_decoder_cross_attn_projk,
                    )
                    lora_cross_attn_projk.weight.data.copy_(
                        block.cross_attn.projk.weight.data
                    )
                    if block.cross_attn.projk.bias is not None:
                        lora_cross_attn_projk.bias.data.copy_(
                            block.cross_attn.projk.bias.data
                        )

                    setattr(dec_blocks[ind].cross_attn, "projk", lora_cross_attn_projk)

                if (
                    hasattr(self.args, "lora_rank_decoder_cross_attn_projv")
                    and self.args.lora_rank_decoder_cross_attn_projv > 0
                ):
                    lora_cross_attn_projv = lora.Linear(
                        block.cross_attn.projv.in_features,
                        block.cross_attn.projv.out_features,
                        r=self.args.lora_rank_decoder_cross_attn_projv,
                        lora_alpha=self.args.lora_alpha_decoder_cross_attn_projv,
                    )
                    lora_cross_attn_projv.weight.data.copy_(
                        block.cross_attn.projv.weight.data
                    )
                    if block.cross_attn.projv.bias is not None:
                        lora_cross_attn_projv.bias.data.copy_(
                            block.cross_attn.projv.bias.data
                        )

                    setattr(dec_blocks[ind].cross_attn, "projv", lora_cross_attn_projv)

                if (
                    hasattr(self.args, "lora_rank_decoder_cross_attn_proj")
                    and self.args.lora_rank_decoder_cross_attn_proj > 0
                ):
                    lora_cross_attn_proj = lora.Linear(
                        block.cross_attn.proj.in_features,
                        block.cross_attn.proj.out_features,
                        r=self.args.lora_rank_decoder_cross_attn_proj,
                        lora_alpha=self.args.lora_alpha_decoder_cross_attn_proj,
                    )
                    lora_cross_attn_proj.weight.data.copy_(
                        block.cross_attn.proj.weight.data
                    )
                    if block.cross_attn.proj.bias is not None:
                        lora_cross_attn_proj.bias.data.copy_(
                            block.cross_attn.proj.bias.data
                        )

                    setattr(dec_blocks[ind].cross_attn, "proj", lora_cross_attn_proj)

                if (
                    hasattr(self.args, "lora_rank_decoder_mlp_fc1")
                    and self.args.lora_rank_decoder_mlp_fc1 > 0
                ):
                    lora_mlp_fc1 = lora.Linear(
                        block.mlp.fc1.in_features,
                        block.mlp.fc1.out_features,
                        r=self.args.lora_rank_decoder_mlp_fc1,
                        lora_alpha=self.args.lora_alpha_decoder_mlp_fc1,
                    )
                    lora_mlp_fc1.weight.data.copy_(block.mlp.fc1.weight.data)
                    if block.mlp.fc1.bias is not None:
                        lora_mlp_fc1.bias.data.copy_(block.mlp.fc1.bias.data)

                    setattr(dec_blocks[ind].mlp, "fc1", lora_mlp_fc1)

                if (
                    hasattr(self.args, "lora_rank_decoder_mlp_fc2")
                    and self.args.lora_rank_decoder_mlp_fc2 > 0
                ):
                    lora_mlp_fc2 = lora.Linear(
                        block.mlp.fc2.in_features,
                        block.mlp.fc2.out_features,
                        r=self.args.lora_rank_decoder_mlp_fc2,
                        lora_alpha=self.args.lora_alpha_decoder_mlp_fc2,
                    )
                    lora_mlp_fc2.weight.data.copy_(block.mlp.fc2.weight.data)
                    if block.mlp.fc2.bias is not None:
                        lora_mlp_fc2.bias.data.copy_(block.mlp.fc2.bias.data)

                    setattr(dec_blocks[ind].mlp, "fc2", lora_mlp_fc2)

    def replace_patch_embed_with_lora(self):
        if (
            hasattr(self.args, "lora_rank_patch_embed_proj")
            and self.args.lora_rank_patch_embed_proj > 0
        ):
            lora_patch_embed = lora.Conv2d(
                in_channels=self.patch_embed.proj.in_channels,
                out_channels=self.patch_embed.proj.out_channels,
                kernel_size=self.patch_embed.proj.kernel_size[0],
                stride=self.patch_embed.proj.stride,
                r=self.args.lora_rank_patch_embed_proj,
                lora_alpha=self.args.lora_alpha_patch_embed_proj,
            )
            lora_patch_embed.conv.weight.data.copy_(self.patch_embed.proj.weight.data)
            if self.patch_embed.proj.bias is not None:
                lora_patch_embed.conv.bias.data.copy_(self.patch_embed.proj.bias.data)

            setattr(self.patch_embed, "proj", lora_patch_embed)

    def replace_head_dpt_with_lora(self):
        for downstream_head in (self.downstream_head1, self.downstream_head2):
            for layer in ["layer1", "layer2", "layer3", "layer4"]:
                if (
                    hasattr(self.args, f"lora_rank_head_dpt_{layer}_rn")
                    and getattr(self.args, f"lora_rank_head_dpt_{layer}_rn") > 0
                ):
                    layer_rn = getattr(downstream_head.dpt.scratch, f"{layer}_rn")
                    # print(f"layer_rn.kernel_size: {layer_rn.kernel_size}, layer_rn.stride: {layer_rn.stride}, layer_rn.padding: {layer_rn.padding}")
                    lora_dpt_scratch_layer = lora.Conv2d(
                        in_channels=layer_rn.in_channels,
                        out_channels=layer_rn.out_channels,
                        kernel_size=layer_rn.kernel_size[0],
                        stride=layer_rn.stride,
                        padding=layer_rn.padding,
                        r=getattr(self.args, f"lora_rank_head_dpt_{layer}_rn"),
                        lora_alpha=getattr(
                            self.args, f"lora_alpha_head_dpt_{layer}_rn"
                        ),
                    )

                    lora_dpt_scratch_layer.conv.weight.data.copy_(layer_rn.weight.data)
                    if layer_rn.bias is not None:
                        lora_dpt_scratch_layer.conv.bias.data.copy_(layer_rn.bias.data)

                    setattr(
                        downstream_head.dpt.scratch,
                        f"{layer}_rn",
                        lora_dpt_scratch_layer,
                    )

            for refinenet in [
                downstream_head.dpt.scratch.refinenet1,
                downstream_head.dpt.scratch.refinenet2,
                downstream_head.dpt.scratch.refinenet3,
                downstream_head.dpt.scratch.refinenet4,
            ]:
                if (
                    hasattr(self.args, "lora_rank_head_dpt_refinenet_out_conv")
                    and self.args.lora_rank_head_dpt_refinenet_out_conv > 0
                ):
                    lora_refinenet_out_conv = lora.Conv2d(
                        in_channels=refinenet.out_conv.in_channels,
                        out_channels=refinenet.out_conv.out_channels,
                        kernel_size=refinenet.out_conv.kernel_size[0],
                        stride=refinenet.out_conv.stride,
                        padding=refinenet.out_conv.padding,
                        r=self.args.lora_rank_head_dpt_refinenet_out_conv,
                        lora_alpha=self.args.lora_alpha_head_dpt_refinenet_out_conv,
                    )
                    lora_refinenet_out_conv.conv.weight.data.copy_(
                        refinenet.out_conv.weight.data
                    )
                    if refinenet.out_conv.bias is not None:
                        lora_refinenet_out_conv.conv.bias.data.copy_(
                            refinenet.out_conv.bias.data
                        )

                    setattr(refinenet, "out_conv", lora_refinenet_out_conv)

                if (
                    hasattr(self.args, "lora_rank_head_dpt_refinenet_conv")
                    and self.args.lora_rank_head_dpt_refinenet_conv > 0
                ):
                    for unit_name in ("resConfUnit1", "resConfUnit2"):
                        resConfUnit = getattr(refinenet, unit_name)
                        for conv_name in ("conv1", "conv2"):
                            conv_block = getattr(resConfUnit, conv_name)
                            lora_conv = lora.Conv2d(
                                in_channels=conv_block.in_channels,
                                out_channels=conv_block.out_channels,
                                kernel_size=conv_block.kernel_size[0],
                                stride=conv_block.stride,
                                padding=conv_block.padding,
                                bias=(conv_block.bias is not None),
                                r=self.args.lora_rank_head_dpt_refinenet_conv,
                                lora_alpha=self.args.lora_alpha_head_dpt_refinenet_conv,
                            )

                            lora_conv.conv.weight.data.copy_(conv_block.weight.data)
                            if conv_block.bias is not None:
                                lora_conv.conv.bias.data.copy_(conv_block.bias.data)

                            setattr(resConfUnit, conv_name, lora_conv)

            for layer in ["0", "2", "4"]:
                if (
                    hasattr(self.args, "lora_rank_head_dpt_head_conv2d")
                    and self.args.lora_rank_head_dpt_head_conv2d > 0
                ):
                    layer_conv = getattr(downstream_head.dpt.head, layer)
                    lora_dpt_head_conv = lora.Conv2d(
                        in_channels=layer_conv.in_channels,
                        out_channels=layer_conv.out_channels,
                        kernel_size=layer_conv.kernel_size[0],
                        stride=layer_conv.stride,
                        padding=layer_conv.padding,
                        r=self.args.lora_rank_head_dpt_head_conv2d,
                        lora_alpha=self.args.lora_alpha_head_dpt_head_conv2d,
                    )
                    lora_dpt_head_conv.conv.weight.data.copy_(layer_conv.weight.data)
                    if layer_conv.bias is not None:
                        lora_dpt_head_conv.conv.bias.data.copy_(layer_conv.bias.data)

                    setattr(downstream_head.dpt.head, layer, lora_dpt_head_conv)

    def load_pretrained_lora(self, ckpt_lora, allow_drop=False):
        metadata_json_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_lora)), "train_encoder_args.json"
        )
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)

        # If this checkpoint was *initialized* from another LoRA (e.g. Dark3R
        # -> Underwat3R cross-domain init), the actual LoRA dims/ranks are
        # determined by the UPSTREAM training, not by the args we recorded
        # at our own training time. Walk one hop up so the LoRA modules we
        # build here have the same shapes as the saved tensors.
        init_lora = metadata.get("init_lora_weights")
        if init_lora and os.path.isfile(init_lora):
            upstream_args = os.path.join(
                os.path.dirname(os.path.dirname(init_lora)), "train_encoder_args.json"
            )
            if os.path.isfile(upstream_args):
                with open(upstream_args, "r") as f:
                    upstream_meta = json.load(f)
                # Inherit ONLY the ENCODER/DECODER LoRA dims — the ckpt's enc/dec
                # LoRA was init'd from the upstream's, so those shapes must match.
                # Do NOT inherit HEAD ranks (head-dpt / head-mlp / local-feature):
                # those are OUR additions on top of the upstream, which records
                # them as -1 (absent). Inheriting -1 skips building the trained
                # DPT-head LoRA -> load_state_dict silently drops ~100 tensors ->
                # depth head runs FROZEN (docs/SWIMM3R_ARCH_ABSORPTION.md #1). Keep
                # our own head ranks so the DPT LoRA modules are rebuilt + loaded.
                _inherited = 0
                for k, v in upstream_meta.items():
                    if not (k.startswith("lora_rank_") or k.startswith("lora_alpha_")):
                        continue
                    if "encoder" in k or "decoder" in k:
                        metadata[k] = v
                        _inherited += 1
                print(
                    f"[load_pretrained_lora] inherited {_inherited} enc/dec LoRA "
                    f"dim(s) from upstream init checkpoint at {upstream_args} "
                    f"(head/DPT ranks kept from this ckpt's own metadata)"
                )

        self.args = argparse.Namespace(**metadata)

        self.replace_encoder_attention_with_lora()
        self.replace_decoder_with_lora()
        self.replace_head_local_feature_mlp_with_lora()
        self.replace_head_dpt_with_lora()

        # weights_only=False because checkpoints may include argparse.Namespace
        # alongside tensors (legacy Dark3R format).
        try:
            sd = torch.load(ckpt_lora, weights_only=True)
        except Exception:
            sd = torch.load(ckpt_lora, weights_only=False)
        self.load_state_dict(sd, strict=False)
        self._verify_lora_landed(sd, allow_drop=allow_drop)
        lora.mark_only_lora_as_trainable(self)

    def _verify_lora_landed(self, sd, allow_drop=False):
        """Guard: confirm every LoRA tensor in the ckpt landed in the model.

        load_state_dict(strict=False) above silently DROPS any ckpt tensor with
        no matching module. The known failure is the DPT-head LoRA: when a ckpt
        was init'd from an upstream LoRA whose DPT ranks were -1, those ranks
        are inherited (see the block above), the DPT LoRA modules are never
        built, and the trained `*.dpt.*.lora_A/lora_B` tensors vanish -> the
        depth head runs FROZEN with no error. Runs on EVERY load path
        (train-init/resume, forward_dark3r eval, swimm3r descatter).

        Dropped DPT (path segment "dpt") tensor -> hard-fail (unless
        allow_drop); dropped non-DPT -> warning; all-landed -> OK log.
        """
        if not isinstance(sd, dict):
            return
        model_keys = set(self.state_dict().keys())
        lora_keys = [k for k in sd
                     if isinstance(k, str) and (".lora_A" in k or ".lora_B" in k)]
        dropped = [k for k in lora_keys if k not in model_keys]
        if not dropped:
            if lora_keys:
                print(f"[lora-verify] OK — all {len(lora_keys)} LoRA tensor(s) "
                      f"landed (incl. DPT-head LoRA if present).")
            return
        def _is_dpt(k):
            return "dpt" in k.split(".")
        dpt_dropped   = [k for k in dropped if _is_dpt(k)]
        other_dropped = [k for k in dropped if not _is_dpt(k)]
        if other_dropped:
            print(f"[lora-verify][WARN] {len(other_dropped)}/{len(lora_keys)} "
                  f"non-DPT LoRA tensor(s) silently dropped, e.g. {other_dropped[:3]}")
        if dpt_dropped:
            msg = (f"[lora-verify] {len(dpt_dropped)} DPT-head LoRA tensor(s) in "
                   f"the ckpt were NOT loaded (silently dropped by strict=False) "
                   f"-> depth head runs FROZEN. Upstream-rank-inheritance bug "
                   f"(docs/SWIMM3R_ARCH_ABSORPTION.md #1). Fix: patch the ckpt's "
                   f"train_encoder_args.json lora_rank_head_dpt_*=<trained rank>. "
                   f"Dropped: {dpt_dropped[:4]}")
            if allow_drop:
                print("[lora-verify][WARN] " + msg + "  — continuing (allow_drop).")
            else:
                raise RuntimeError(msg + "  — set allow_drop / --allow_lora_drop to override.")

    def setup_for_decoder_only(self, ckpt_lora, args):
        # metadata_json_path = ckpt_lora.replace(".pth", ".json")
        metadata_json_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_lora)), "train_encoder_args.json"
        )
        with open(metadata_json_path, "r") as f:
            metadata = json.load(f)
            self.args = argparse.Namespace(**metadata)

        self.replace_encoder_attention_with_lora()
        self.load_state_dict(torch.load(ckpt_lora), strict=False)
        self.args = args
        self.replace_decoder_with_lora(self.args)
        lora.mark_only_lora_as_trainable(self)

    def save_lora_weights(self, checkpoint_path):
        torch.save(lora.lora_state_dict(self), checkpoint_path)

    def _encode_image(self, image, true_shape):
        return super()._encode_image(image, true_shape)
