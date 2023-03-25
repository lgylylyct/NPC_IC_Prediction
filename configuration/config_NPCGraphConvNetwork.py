class Configuration:
    def __init__(self) -> None:
        pass

    def to_dict(self):
        return self.__dict__


cfg_gcn = Configuration()


cfg_gcn.clinic_channels = 14
cfg_gcn.schedule_cat_clinic = None
cfg_gcn.dropout = 0.5

cfg_gcn.add_pdt = False
cfg_gcn.pdc_hidden_dims = 320
cfg_gcn.pdc_num_layers = 2
cfg_gcn.pdc_norm = "bn"
cfg_gcn.pdc_atc = "relu"
cfg_gcn.pdc_last_norm = None

cfg_gcn.n_classes = 1
cfg_gcn.k = 16
cfg_gcn.block = "plain"
cfg_gcn.conv = "mr"
cfg_gcn.act = "relu"
cfg_gcn.norm = None
cfg_gcn.bias = True
cfg_gcn.n_filters = 64
cfg_gcn.n_blocks = 10

# dilated knn
cfg_gcn.grow_num = 0.0
cfg_gcn.epsilon = 0.2
cfg_gcn.stochastic = True

cfg_gcn.init_strategy = None
