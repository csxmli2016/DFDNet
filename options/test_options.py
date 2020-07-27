from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # parser.add_argument('--dataroot', type=str, default='/home/Data/AllDataImages/2018_FaceFH', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--dataroot', type=str, default='', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--phase', type=str, default='', help='train, val, test, etc')
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--is_real', type=int, default=0, help='0:for test with degradation, 1: for real without degradation')
        parser.add_argument('--partroot', type=str, default='datasets/GRMouthVGG2/MergeGRVGG2WebFace/', help='path to roialign part locations')

        parser.set_defaults(model='test')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))


        parser.add_argument('--p1', type=str, default='', help='')
        parser.add_argument('--p2', type=str, default='', help='')
        parser.add_argument('--p3', type=str, default='', help='')
        parser.add_argument('--p4', type=int, default=0, help='')
        parser.add_argument('--p5', type=str, default='', help='')
        parser.add_argument('--p6', type=str, default='', help='')
        
        self.isTrain = False
        return parser
