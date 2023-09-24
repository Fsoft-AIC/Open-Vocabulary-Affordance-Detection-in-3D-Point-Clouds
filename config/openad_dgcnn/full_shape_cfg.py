import os
from os.path import join as opj

exp_name = "OPENAD_DGCNN_FULL_SHAPE_Release"
work_dir = opj("./log/openad_dgcnn", exp_name)
seed = 1
try:
    os.makedirs(work_dir)
except:
    print('Working Dir is already existed!')

scheduler = dict(
    type='cos',
    T_max=200,
    eta_min=1e-3
)
optimizer = dict(
    type='sgd',
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4
)

model = dict(
    type='openad_dgcnn',
    k=40,
    emb_dims=1024
)

training_cfg = dict(
    model=model,
    estimate=True,
    partial=False,
    rotate='None',
    semi=False,
    rotate_type=None,
    batch_size=16,
    epoch=200,
    seed=1,
    dropout=0.5,
    gpu='4',
    workflow=dict(
        train=1,
        val=1
    ),
    train_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    val_affordance = ['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
               'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
               'listen', 'wear', 'press', 'cut', 'stab', 'none'],
    weights_dir = './data/full_shape_weights.npy'
)

data = dict(
    data_root='./data',
    category=['grasp', 'contain', 'lift', 'openable', 'layable', 'sittable',
              'support', 'wrap_grasp', 'pourable', 'move', 'displaY', 'pushable', 'pull',
              'listen', 'wear', 'press', 'cut', 'stab', 'none']
)