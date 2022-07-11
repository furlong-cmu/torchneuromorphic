from torchneuromorphic.dvs_gestures.dvsgestures_dataloaders import *
import snntorch.spikeplot as splt
import matplotlib.pyplot as plt

train_dl, test_dl = create_dataloader(
    root='/Users/ZP/Downloads/DVSbuild/dvs_gestures_build19.hdf5',
    batch_size=32,
    ds=4,
    num_workers=0)

# data = iter(train_dl)
# data_it = next(data._sampler_iter)
fig, ax = plt.subplots()

# for i, v in enumerate(train_dl):
#     print(np.argmax(v[1][0]))
#     tt = v[0][0,:,1,:,:]
#     anim = splt.animator(tt, fig, ax)
#     plt.show()
#     break

pass
