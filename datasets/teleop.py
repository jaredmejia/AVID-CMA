import os
import glob
import numpy as np
import pandas as pd
import torch
import torchaudio

from PIL import Image

DATA_PATH = "/home/vdean/franka_demo/logs/jared_chopping_exps_v3/pos_1/"
CONTACT_AUDIO_FREQ = 32000

from datasets.video_db import VideoDataset

DEBUG = False


class Teleop(VideoDataset):
    def __init__(
        self,
        subset,
        return_video=True,
        video_clip_duration=1.0,
        video_fps=25.0,
        video_transform=None,
        return_audio=False,
        audio_clip_duration=1.0,
        audio_fps=None,
        audio_fps_out=64,
        audio_transform=None,
        return_labels=False,
        return_index=False,
        max_offsync_augm=0,
        mode="clip",
        clips_per_video=1,
    ):
        classes = []
        filenames = get_filenames(DATA_PATH)
        video_fns = [f"{fn_list[0]}.jpeg" for fn_list in filenames]
        audio_fns = [f"{fn_list[0]}.txt" for fn_list in filenames]
        labels = []

        super(Teleop, self).__init__(
            return_video=return_video,
            video_root=f"{DATA_PATH}",
            video_fns=video_fns,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=f"{DATA_PATH}",
            audio_fns=audio_fns,
            audio_clip_duration=audio_clip_duration,
            audio_fps=audio_fps,
            audio_fps_out=audio_fps_out,
            audio_transform=audio_transform,
            return_labels=return_labels,
            labels=labels,
            return_index=return_index,
            mode=mode,
            clips_per_video=clips_per_video,
            max_offsync_augm=max_offsync_augm,
        )

        self.name = "Teleop dataset"
        self.root = DATA_PATH
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)
        self.filenames = filenames

        # self.sample_id = np.array(
        #     [fn.split("/")[-1].split(".")[0].encode("utf-8") for fn in filenames]
        # )
        self.audio_resampler = torchaudio.transforms.Resample(
            orig_freq=CONTACT_AUDIO_FREQ, new_freq=audio_fps, dtype=torch.float64
        )

    def __getitem__(self, idx):
        sample = {}

        if self.return_video:
            frames = []
            for base_path in self.filenames[idx]:
                img_path = f"{base_path}.jpeg"
                img = Image.open(img_path)
                frames.append(img)

            if self.video_transform is not None:
                for t in self.video_transform:
                    frames = t(frames)

            sample["frames"] = frames

            if DEBUG:
                import matplotlib.pyplot as plt

                f, ax = plt.subplots(1, 8, figsize=(20, 5))
                for i in range(8):
                    ax[i].imshow(sample["frames"][:, i, :, :].permute(1, 2, 0))
                f.savefig("./frames_fig.png")
                input("Press Enter to continue")

        if self.return_audio:
            audio_list = []
            for base_path in self.filenames[idx]:
                audio_path = f"{base_path}.txt"
                txt_arr = np.loadtxt(audio_path)
                txt_arr = txt_arr.T
                audio_list.append(txt_arr)

            audio_data = np.concatenate(audio_list, axis=1)

            # audio resampling
            audio = torch.tensor(audio_data)
            audio = self.audio_resampler(audio)
            audio = audio.numpy().astype(np.int16)
            audio = audio / np.iinfo(audio.dtype).max
            samples, rate = audio, self.audio_fps

            if self.audio_transform is not None:
                if isinstance(self.audio_transform, list):
                    for t in self.audio_transform:
                        # using video_clip_duration here since audio is exactly corresponding to video frames
                        # NOTE:  may change this later to audio_clip_duration
                        samples, rate = t(samples, rate, self.video_clip_duration)
                else:
                    samples, rate = self.audio_transform(samples, rate)

            sample["audio"] = samples

            if DEBUG:
                plt.clf()
                # plt.specgram(sample["audio"][0])
                plt.imshow(sample["audio"][0], cmap="hot", interpolation="nearest")
                plt.savefig("./spec-fig.png")

                plt.clf()
                for i in range(4):
                    plt.plot(list(range(audio.shape[1])), audio[i])
                plt.savefig("./audio-fig.png")

        if self.return_index:
            sample["index"] = idx

        return sample

    def __repr__(self):
        desc = "{}\n - Root: {}\n - Subset: {}\n - Num videos: {}\n - Num samples: {}\n".format(
            self.name,
            self.root,
            self.subset,
            self.num_videos,
            self.num_videos * self.clips_per_video,
        )
        if self.return_video:
            desc += " - Example video: {}/{}.jpeg\n".format(
                self.video_root, self.video_fns[0].decode()
            )
        if self.return_audio:
            desc += " - Example audio: {}/{}.txt\n".format(
                self.audio_root, self.audio_fns[0].decode()
            )
        return desc


def get_filenames(root, num_images_cat=8):
    dir_names = [txt_fn[:-4] for txt_fn in glob.glob(os.path.join(root, "*.txt"))]
    filenames = []
    for dir_name in dir_names:
        sub_files = [
            img_fn[:-5]
            for img_fn in sorted(glob.glob(os.path.join(dir_name, "*.jpeg")))
        ]
        for i in range(num_images_cat, len(sub_files), num_images_cat):
            sub_files_cat = sub_files[i - num_images_cat : i]
            filenames.append(sub_files_cat)
    return filenames
