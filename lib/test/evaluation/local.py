from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/home/long/mixformerv2-extract-information-from-video/data/got10k_lmdb'
    settings.got10k_path = '/home/long/mixformerv2-extract-information-from-video/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/home/long/mixformerv2-extract-information-from-video/data/lasot_lmdb'
    settings.lasot_path = '/home/long/mixformerv2-extract-information-from-video/data/lasot'
    settings.network_path = '/home/long/mixformerv2-extract-information-from-video/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/home/long/mixformerv2-extract-information-from-video/data/nfs'
    settings.otb_path = '/home/long/mixformerv2-extract-information-from-video/data/OTB2015'
    settings.prj_dir = '/home/long/mixformerv2-extract-information-from-video'
    settings.result_plot_path = '/home/long/mixformerv2-extract-information-from-video/test/result_plots'
    settings.results_path = '/home/long/mixformerv2-extract-information-from-video/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/home/long/mixformerv2-extract-information-from-video'
    settings.segmentation_path = '/home/long/mixformerv2-extract-information-from-video/test/segmentation_results'
    settings.tc128_path = '/home/long/mixformerv2-extract-information-from-video/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/home/long/mixformerv2-extract-information-from-video/data/trackingNet'
    settings.uav_path = '/home/long/mixformerv2-extract-information-from-video/data/UAV123'
    settings.vot_path = '/home/long/mixformerv2-extract-information-from-video/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

