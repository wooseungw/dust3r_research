#데이터셋 예제
from dust3r.datasets.base.base_stereo_view_dataset import view_name
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb
from dust3r.datasets.co3d import Co3d
import numpy as np
#모델
from dust3r.model import AsymmetricCroCo3DStereo 

model = AsymmetricCroCo3DStereo(patch_embed_cls='ManyAR_PatchEmbed')
dataset = Co3d(split='train', ROOT="data/co3d_subset_processed", resolution=224, aug_crop=16)

for idx in np.random.permutation(len(dataset)):
    views = dataset[idx]
    assert len(views) == 2
    print(view_name(views[0]), view_name(views[1]))
    viz = SceneViz()
    poses = [views[view_idx]['camera_pose'] for view_idx in [0, 1]]
    cam_size = max(auto_cam_size(poses), 0.001)
    for view_idx in [0, 1]:
        pts3d = views[view_idx]['pts3d']
        valid_mask = views[view_idx]['valid_mask']
        colors = rgb(views[view_idx]['img'])
        viz.add_pointcloud(pts3d, colors, valid_mask)
        viz.add_camera(pose_c2w=views[view_idx]['camera_pose'],
                        focal=views[view_idx]['camera_intrinsics'][0, 0],
                        color=(idx*255, (1 - idx)*255, 0),
                        image=colors,
                        cam_size=cam_size)
    viz.show()

