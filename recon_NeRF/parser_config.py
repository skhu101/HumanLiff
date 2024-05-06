import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--ddp", type=int, default=0)

    # training options
    parser.add_argument("--n_iteration",   type=int, default=50000, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--tri_plane_lrate", type=float, default=1e-3, 
                        help='learning rate')
    parser.add_argument("--decay_steps", type=int, default=10000, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--acc_loss", type=int, default=1, help="--- ")
    parser.add_argument("--use_canonical_space", action='store_true', help="--- ")
    parser.add_argument("--use_clamp", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--mlp_num", type=int, default=8, help="--- ")
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--tv_loss", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--tv_loss_coef", type=float, default=5e-4, 
                        help='tv loss coef')
    parser.add_argument("--l1_loss_coef", type=float, default=2e-4, 
                        help='l1 loss coef')
    parser.add_argument("--normal_loss_coef", type=float, default=1e-2, 
                        help='normal loss coef')
    parser.add_argument("--split", action='store_true', 
                        help='split alpha and color network')

    # triplane options
    parser.add_argument("--triplane_loss", action='store_true', help="--- ")
    parser.add_argument("--triplane_dim", type=int, default=256, help="--- ")
    parser.add_argument("--triplane_ch", type=int, default=18, help="--- ")
    parser.add_argument("--start_dim", type=int, default=64, help="--- ")

    # rendering options
    parser.add_argument("--n_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--n_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--n_rand", type=int, default=1024*32, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--chunk", type=int, default=1024*64, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--with_viewdirs", type=int, default=1, 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--raw_noise_std", type=float, default=1, 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--sample_npz", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--ft_triplane_only", action='store_true', 
                        help='only fit triplane')

    # dataset options
    parser.add_argument("--data_root", type=str, default='msra_h36m/S9/Posing', 
                        help='Dataset root dir')
    parser.add_argument("--data_set_type", type=str, default='multi_pair', 
                        help='Dataset root dir')
    parser.add_argument("--train_split", type=str, default="test", 
                        help='training dataloader type, choose whole image or random sample')
    parser.add_argument("--test_split", type=str, default="test", 
                        help='test dataloader type, choose whole image or random sample')
    parser.add_argument("--image_scaling", type=float, default="1.0", 
                        help='down sample factor')
    parser.add_argument("--model", type=str, default="correction_by_f3d", 
                        help='test dataloader type, choose whole image or random sample')
    parser.add_argument("--num_worker", type=int, default=8, help="--- ")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--multi_person", type=int, default=1, help="--- ")
    parser.add_argument("--start", type=int, default=0, help="--- ")
    parser.add_argument("--interval", type=int, default=10, help="--- ")
    parser.add_argument("--poses_num", type=int, default=100, help="--- ")
    parser.add_argument("--views_num", type=int, default=382, help="--- ")
    parser.add_argument("--num_instance", type=int, default=100, help="--- ")
    parser.add_argument("--occupancy", type=int, default=0, help="--- ")
    parser.add_argument("--start_idx", type=int, default=0, help="--- ")
    parser.add_argument("--end_idx", type=int, default=762, help="--- ")

    # logging/saving/testing options
    parser.add_argument("--test", action='store_true', help="--- ")
    parser.add_argument("--test_layer_id", type=int, default=-1, help="--- ")
    parser.add_argument("--i_print",   type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=10000, help='frequency of testset saving')

    return parser


def print_args(args):
    print('--------args----------')
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('--------args----------\n')