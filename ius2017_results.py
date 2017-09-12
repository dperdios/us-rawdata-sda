import os
import tensorflow as tf
import copy
import numpy as np
import pickle
from skimage.measure import compare_psnr, compare_ssim
from lip.model import SDA
import datasets.utils as dutils
import networks.utils as nutils

# Load PICMUS test set
test_set = dutils.load_ius2017_test_set()

# Saved networks base directory
base_dir = os.path.join('networks', 'ius2017')

# Export folder
export_path = os.path.join('results', 'ius2017')
if not os.path.exists(export_path):
    os.makedirs(export_path)

# Recover data, beamform image and evaluate metrics for trained network on each test case
net_configs = nutils.check_trained_networks(path=base_dir)
data_dim = test_set[0].sample_numbers[0]

for seq_ref in test_set:
    # Beamform reference image
    im_ref = seq_ref.beamform(interp='cubic')

    # Extract envelope
    env_ref = im_ref.envelope(normalize=None)
    env_ref_max = env_ref.max()

    # Compute bmode, normalized on the max of the reference image (envelope)
    bmode_ref = im_ref.bmode(normalize=env_ref_max)

    # Compress bmode and convert to int8
    if '_expe_' in seq_ref.name:
        db_range = 40  # 40dB for in vivo acquisitions
    else:
        db_range = 60  # 60dB for in vitro and numerical acquisitions
    bmode_ref_int8 = (np.where(bmode_ref < -db_range, -db_range, bmode_ref) + db_range) / db_range * 255
    bmode_ref_int8 = np.where(bmode_ref_int8 > 255, 255, bmode_ref_int8).astype(np.uint8)

    # Export reference image
    im_ref_path = os.path.join(export_path, seq_ref.name + '_ref.pdf')
    im_ref.plot_bmode(normalize=env_ref_max, db_range=db_range, cbar=True, save_path=im_ref_path)

    # Evaluate metrics on each trained network
    metrics_list = []
    for cfg in net_configs:
        model = SDA(data_dim=data_dim, base_dir=base_dir, **cfg)
        print('Evaluating {} on {}'.format(model.name, seq_ref.name))
        data_ref = seq_ref.data[0]
        with tf.Session() as sess:
            # Load trained network
            model.restore(sess=sess)

            # Forward pass in network
            data_rec = sess.run(model.outputs, feed_dict={model.inputs: data_ref})

            # Data PSNR and SSIM
            data_psnr = compare_psnr(data_ref, data_rec, data_range=2)
            data_ssim = compare_ssim(data_ref, data_rec, data_range=2, gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False)

            # Beamform image from recovered data
            seq_rec = copy.deepcopy(seq_ref)
            seq_rec.update_data([data_rec])
            im_rec = seq_rec.beamform(interp='cubic')

            # Compute metrics w.r.t. the reference image
            #   Extract envelope
            env_ref = im_ref.envelope(normalize=None)
            env_rec = im_rec.envelope(normalize=None)
            env_ref_max = env_ref.max()
            #   Compute bmode, normalized on the max of the reference image (envelope)
            bmode_rec = im_rec.bmode(normalize=env_ref_max)
            #   Compress bmode and convert to int8
            bmode_rec_int8 = (np.where(bmode_rec < -db_range, -db_range, bmode_rec) + db_range) / db_range * 255
            bmode_rec_int8 = (np.where(bmode_rec_int8 > 255, 255, bmode_rec_int8)).astype(np.uint8)
            #   Compute metrics on the compressed bmode images
            bmode_psnr = compare_psnr(bmode_ref_int8, bmode_rec_int8, data_range=255)
            bmode_ssim = compare_ssim(bmode_ref_int8, bmode_rec_int8, data_range=255,
                                      gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

            # Display some metrics
            print('Test case: {}'.format(seq_ref.name))
            print('  Network name: {}'.format(model.name))
            print('  Data PSNR: {:.2f}'.format(data_psnr))
            print('  Data SSIM: {:.2f}'.format(data_ssim))
            print('  Bmode PSNR: {:.2f}'.format(bmode_psnr))
            print('  Bmode SSIM: {:.2f}'.format(bmode_ssim))

            # Save metrics
            model_metrics = dict()
            model_metrics['model_name'] = model.name
            model_metrics['data_psnr'] = data_psnr
            model_metrics['data_ssim'] = data_ssim
            model_metrics['bmode_psnr'] = bmode_psnr
            model_metrics['bmode_ssim'] = bmode_ssim
            model_metrics['model_config'] = cfg
            metrics_list.append(model_metrics)

            # Export figures
            im_base_path = os.path.join(export_path, '_'.join([seq_ref.name, model.name]))
            im_rec_path = os.path.join(export_path, '_'.join([seq_ref.name, model.name])) + '_rec.pdf'
            im_rec_path = im_rec_path.replace('-', '_')
            im_rec.plot_bmode(normalize=env_ref_max, db_range=db_range, save_path=im_rec_path, cbar=True)

    # Save metrics on all models for the test case
    metrics_path = os.path.join(export_path, seq_ref.name + '_metrics.pickle')
    with open(metrics_path, 'wb') as handle:
        pickle.dump(metrics_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
