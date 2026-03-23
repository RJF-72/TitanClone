# backend_logic.py

import logging
import os
import soundfile
import subprocess
import torch

from inference import infer_tool
from inference.infer_tool import Svc
from spkmix import spk_mix_map

# Configure logging
logging.getLogger('numba').setLevel(logging.WARNING)

class TitanCloneBackend:
    def __init__(self):
        self.svc_model = None

    def load_inference_model(self, model_path, config_path, device=None, cluster_model_path="",
                             enhance=False, diffusion_model_path="", diffusion_config_path="",
                             shallow_diffusion=False, only_diffusion=False, use_spk_mix=False,
                             feature_retrieval=False):
        try:
            self.svc_model = Svc(model_path, config_path, device,
                                 cluster_model_path, enhance, diffusion_model_path,
                                 diffusion_config_path, shallow_diffusion, only_diffusion,
                                 use_spk_mix, feature_retrieval)
            return {"status": "success", "message": "Inference model loaded successfully."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to load inference model: {e}"}

    def run_inference(self, raw_audio_path, spk, tran, slice_db, cluster_infer_ratio,
                      auto_predict_f0, noice_scale, pad_seconds, clip, lg, lgr, f0p,
                      enhancer_adaptive_key, cr_threshold, k_step, use_spk_mix,
                      second_encoding, loudness_envelope_adjustment, wav_format="flac"):
        if not self.svc_model:
            return {"status": "error", "message": "Inference model not loaded. Please load a model first."}
        
        try:
            # Ensure output directories exist
            infer_tool.mkdir(["raw", "results"])

            # This part handles single file inference as per inference_main.py
            # For simplicity, we'll assume raw_audio_path is already prepared or can be handled.
            # In a real UI, you'd handle file uploads and saving to 'raw' folder.

            # Example of processing raw_audio_path if it's just a filename
            # You might need to adjust this based on how the frontend sends the file.
            full_raw_audio_path = os.path.join("raw", os.path.basename(raw_audio_path))
            if not os.path.exists(full_raw_audio_path):
                # In a real app, this would be a file uploaded from the UI.
                # For now, let's just use the provided raw_audio_path directly,
                # assuming it's an absolute path or relative to the CWD.
                pass
            
            infer_tool.format_wav(raw_audio_path)

            # kwarg construction from inference_main.py
            kwarg = {
                "raw_audio_path" : raw_audio_path,
                "spk" : spk,
                "tran" : tran,
                "slice_db" : slice_db,
                "cluster_infer_ratio" : cluster_infer_ratio,
                "auto_predict_f0" : auto_predict_f0,
                "noice_scale" : noice_scale,
                "pad_seconds" : pad_seconds,
                "clip_seconds" : clip,
                "lg_num": lg,
                "lgr_num" : lgr,
                "f0_predictor" : f0p,
                "enhancer_adaptive_key" : enhancer_adaptive_key,
                "cr_threshold" : cr_threshold,
                "k_step":k_step,
                "use_spk_mix":use_spk_mix,
                "second_encoding":second_encoding,
                "loudness_envelope_adjustment":loudness_envelope_adjustment
            }
            
            audio = self.svc_model.slice_inference(**kwarg)

            key = "auto" if auto_predict_f0 else f"{tran}key"
            cluster_name = "" if cluster_infer_ratio == 0 else f"_{cluster_infer_ratio}"
            isdiffusion = "sovits"
            if shallow_diffusion :
                isdiffusion = "sovdiff"
            if only_diffusion :
                isdiffusion = "diff"
            # This part about spk_mix needs careful handling if use_spk_mix is True
            # For now, assuming spk is a string, not a dict from spk_mix_map
            # if use_spk_mix: spk = "spk_mix" # This logic needs spk_list from args

            clean_name = os.path.basename(raw_audio_path)
            res_path = f'results/{clean_name}_{key}_{spk}{cluster_name}_{isdiffusion}_{f0p}.{wav_format}'
            soundfile.write(res_path, audio, self.svc_model.target_sample, format=wav_format)
            self.svc_model.clear_empty()

            return {"status": "success", "message": "Inference completed successfully.", "output_path": res_path}
        except Exception as e:
            return {"status": "error", "message": f"Inference failed: {e}"}

    def run_command_and_stream_output(self, command_args):
        # This function is adapted from the Gradio implementation for streaming command output
        process = subprocess.Popen(command_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in process.stdout:
            yield line.strip()
        process.wait()
        if process.returncode != 0:
            yield f"Error: Command exited with code {process.returncode}"

    # Placeholder for training related functions
    def preprocess_dataset(self, dataset_path, speech_encoder, vol_aug=False, num_processes=1):
        # Example: call preprocess_flist_config.py
        command_args = [
            "python", "preprocess_flist_config.py",
            f"--speech_encoder={speech_encoder}"
        ]
        if vol_aug: 
            command_args.append("--vol_aug")
        if num_processes > 1:
            command_args.append(f"--num_processes={num_processes}")
        
        yield from self.run_command_and_stream_output(command_args)

    def generate_hubert_f0(self, f0_predictor, use_diff=False, num_processes=1):
        # Example: call preprocess_hubert_f0.py
        command_args = [
            "python", "preprocess_hubert_f0.py",
            f"--f0_predictor={f0_predictor}"
        ]
        if use_diff:
            command_args.append("--use_diff")
        if num_processes > 1:
            command_args.append(f"--num_processes={num_processes}")

        yield from self.run_command_and_stream_output(command_args)

    def train_model(self, config_path, model_name):
        # Example: call train.py
        command_args = [
            "python", "train.py",
            f"-c", config_path, # Assuming config.json is generated/exists
            f"-m", model_name # e.g., 44k
        ]
        yield from self.run_command_and_stream_output(command_args)


