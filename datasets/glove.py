import os
import glob
import numpy as np
import pandas as pd

DATA_PATH = "/home/vdean/jared_contact_mic/data/glove/16fps_360p_48khz"
META_DATA_PATH = "/home/vdean/jared_contact_mic/avid-glove/datasets/assets/glove"


from datasets.video_db import VideoDataset


class Glove(VideoDataset):
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

        classes = sorted(os.listdir(f"{DATA_PATH}"))
        filenames = get_fnames(DATA_PATH, META_DATA_PATH, subset)
        labels = [classes.index(fn.split("/")[-2]) for fn in filenames]

        super(Glove, self).__init__(
            return_video=return_video,
            video_root=f"{DATA_PATH}",
            video_fns=filenames,
            video_clip_duration=video_clip_duration,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=return_audio,
            audio_root=f"{DATA_PATH}",
            audio_fns=filenames,
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

        self.name = "Glove dataset"
        self.root = DATA_PATH
        self.subset = subset

        self.classes = classes
        self.num_videos = len(filenames)
        self.num_classes = len(classes)

        self.sample_id = np.array(
            [fn.split("/")[-1].split(".")[0].encode("utf-8") for fn in filenames]
        )


def get_fnames(data_dir, meta_dir, subset):
    csv_fn = os.path.join(meta_dir, f"{subset}.csv")
    meta_df = pd.read_csv(csv_fn, header=None)
    fnames = meta_df[0].to_list()
    # fnames = [os.path.join(data_dir, fname) for fname in fnames]
    return fnames
