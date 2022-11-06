# https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html
# https://chrsmrrs.github.io/datasets/docs/datasets/

import torch_geometric.datasets as ds

TUNames = [
    "AIDS",
    "alchemy_full",
    "aspirin",
    "benzene",
    "BZR",
    "BZR_MD",
    "COX2",
    "COX2_MD",
    "DHFR",
    "DHFR_MD",
    "ER_MD",
    "ethanol",
    "FRANKENSTEIN",
    "malonaldehyde",
    "MCF-7",
    "MCF-7H",
    "MOLT-4",
    "MOLT-4H",
    "Mutagenicity",
    "MUTAG",
    "naphthalene",
    "NCI1",
    "NCI109",
    "NCI-H23",
    "NCI-H23H",
    "OVCAR-8",
    "OVCAR-8H",
    "P388",
    "P388H",
    "PC-3",
    "PC-3H",
    "PTC_FM",
    "PTC_FR",
    "PTC_MM",
    "PTC_MR",
    "QM9",
    "salicylic_acid",
    "SF-295",
    "SF-295H",
    "SN12C",
    "SN12CH",
    "SW-620",
    "SW-620H",
    "toluene",
    "Tox21_AhR_training",
    "Tox21_AhR_testing",
    "Tox21_AhR_evaluation",
    "Tox21_AR_training",
    "Tox21_AR_testing",
    "Tox21_AR_evaluation",
    "Tox21_AR-LBD_training",
    "Tox21_AR-LBD_testing",
    "Tox21_AR-LBD_evaluation",
    "Tox21_ARE_training",
    "Tox21_ARE_testing",
    "Tox21_ARE_evaluation",
    "Tox21_aromatase_training",
    "Tox21_aromatase_testing",
    "Tox21_aromatase_evaluation",
    "Tox21_ATAD5_training",
    "Tox21_ATAD5_testing",
    "Tox21_ATAD5_evaluation",
    "Tox21_ER_training",
    "Tox21_ER_testing",
    "Tox21_ER_evaluation",
    "Tox21_ER-LBD_training",
    "Tox21_ER-LBD_testing",
    "Tox21_ER-LBD_evaluation",
    "Tox21_HSE_training",
    "Tox21_HSE_testing",
    "Tox21_HSE_evaluation",
    "Tox21_MMP_training",
    "Tox21_MMP_testing",
    "Tox21_MMP_evaluation",
    "Tox21_p53_training",
    "Tox21_p53_testing",
    "Tox21_p53_evaluation",
    "Tox21_PPAR-gamma_training",
    "Tox21_PPAR-gamma_testing",
    "Tox21_PPAR-gamma_evaluation",
    "UACC257",
    "UACC257H",
    "uracil",
    "Yeast",
    "YeastH",
    "ZINC_full",
    "ZINC_test",
    "ZINC_train",
    "ZINC_val",
    # "Bioinformatics",
    "DD",
    "ENZYMES",
    "KKI",
    "OHSU",
    "Peking_1",
    "PROTEINS",
    "PROTEINS_full",
    # "Computervision",
    "COIL-DEL",
    "COIL-RAG",
    "Cuneiform",
    "Fingerprint",
    "FIRSTMM_DB",
    "Letter-high",
    "Letter-low",
    "Letter-med",
    "MSRC_9",
    "MSRC_21",
    "MSRC_21C",
    # "Socialnetworks",
    "COLLAB",
    "dblp_ct1",
    "dblp_ct2",
    "DBLP_v1",
    "deezer_ego_nets",
    "facebook_ct1",
    "facebook_ct2",
    "github_stargazers",
    "highschool_ct1",
    "highschool_ct2",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "infectious_ct1",
    "infectious_ct2",
    "mit_ct1",
    "mit_ct2",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
    "REDDIT-MULTI-12K",
    "reddit_threads",
    "tumblr_ct1",
    "tumblr_ct2",
    "twitch_egos",
    "TWITTER-Real-Graph-Partial",
    # "Synthetic",
    "COLORS-3",
    "SYNTHETIC",
    "SYNTHETICnew",
    "Synthie",
    "TRIANGLES",

]

# ds.KarateClub()
# for name in TUNames:
#     try:
#         ds.TUDataset(root='/tmp/TUDataset/', name=name)
#     except Exception as e:
#         print(f"TUDataset ... missing {name}", e)
#         pass
#
# for name in ["PATTERN", "CLUSTER", "MNIST", "CIFAR10", "TSP", "CSL"]:
#     ds.GNNBenchmarkDataset(root='/tmp/GNNBenchmarkDataset/', name=name)
#
# for name in ["Cora", "CiteSeer", "PubMed"]:
#     ds.Planetoid(root='/tmp/Planetoid/', name=name)
#
# ds.FakeDataset()
# ds.FakeHeteroDataset()
# ds.NELL(root='/tmp/NELL/')
#
# for name in ["Cora", "Cora_ML" "CiteSeer", "DBLP", "PubMed"]:
#     try:
#         ds.CitationFull(root='/tmp/CitationFull/', name=name)
#     except Exception as e:
#         print(f"CitationFull ... missing {name}", e)
#     pass
#
# ds.CoraFull(root='/tmp/CoraFull/')
#
# for name in["CS", "Physics"]:
#     try:
#         ds.Coauthor(root='/tmp/Coauthor/', name=name)
#     except Exception as e:
#         print(f"Coauthor ... missing {name}", e)
#     pass
#
# for name in ["Computers", "Photo"]:
#     try:
#         ds.Amazon(root='/tmp/Amazon/', name=name)
#     except Exception as e:
#         print(f"Amazon ... missing {name}", e)
#     pass
#
# ds.PPI(root='/tmp/PPI/')
# ds.Reddit(root='/tmp/Reddit/')
# ds.Reddit2(root='/tmp/Reddit2/')
# ds.Flickr(root='/tmp/Flickr/')
# ds.Yelp(root='/tmp/Yelp/')
# ds.AmazonProducts(root='/tmp/AmazonProducts/')
# ds.QM7b(root='/tmp/QM7b/')
# ds.QM9(root='/tmp/QM9/')
#
# for name in ['benzene', 'uracil', 'napthalene', 'aspirin', 'malonaldehyde', 'ethanol', 'toluene', 'paracetamol', 'azobenzene']:
#     try:
#         ds.MD17(root='/tmp/MD17/', name=name)
#     except Exception as e:
#         print(f"MD17 ... missing {name}", e)
#     pass
#
# ds.ZINC(root='/tmp/ZINC/')
#
# for name in ["ESOL", "FreeSolv", "Lipo", "PCBA", "MUV", "HIV", "BACE", "BBPB", "Tox21", "ToxCast", "SIDER", "ClinTox"]:
#     try:
#         ds.MoleculeNet(root='/tmp/MoleculeNet/', name=name)
#     except Exception as e:
#         print(f"MoleculeNet ... missing {name}", e)
#     pass
#
# for name in ["AIFB", "MUTAG", "BGS", "AM"]:
#     try:
#         ds.Entities(root='/tmp/Entities/', name=name)
#     except Exception as e:
#         print(f"Entities ... missing {name}", e)
#     pass
#
# for name in ["FB15k-237"]:
#     ds.RelLinkPredDataset(root='/tmp/RelLinkPredDataset/', name=name)
#
# for name in ["AIDS700nef", "LINUX", "ALKANE", "IMDBMulti"]:
#     try:
#         ds.GEDDataset(root='/tmp/GEDDataset/', name=name)
#     except Exception as e:
#         print(f"GEDDataset ... missing {name}", e)
#     pass
#
# for name in ["Wiki", "Cora" "CiteSeer", "PubMed", "BlogCatalog", "PPI", "Flickr", "Facebook", "Twitter", "TWeibo", "MAG"]:
#     try:
#         ds.AttributedGraphDataset(root='/tmp/AttributedGraphDataset/', name=name)
#     except Exception as e:
#         print(f"AttributedGraphDataset ... missing {name}", e)
#     pass
#
# ds.MNISTSuperpixels(root='/tmp/MNISTSuperpixels/')
# catyaa@hotmail.com/ 9NtXgackschV-a:
try:
    ds.FAUST(root='/tmp/FAUST/')
except Exception as e:
    print(f"FAUST ... missing", e)
try:
    ds.DynamicFAUST(root='/tmp/DynamicFAUST/')
except Exception as e:
    print(f"DynamicFAUST ... missing", e)

ds.ShapeNet(root='/tmp/ShapeNet/')

for name in ['10', '40']:
    ds.ModelNet(root='/tmp/ModelNet/', name=name)

ds.CoMA(root='/tmp/CoMA/')

for partiality in ["Holes", "Cuts"]:
    for category in ["Cat", "Centaur", "David", "Dog", "Horse", "Michael", "Victoria", "Wolf"]:
        try:
            ds.SHREC2016(root='/tmp/SHREC2016/', partiality=partiality, category=category)
        except Exception as e:
            print(f"SHREC2016 ... missing {partiality},{category}", e)
        pass

for category in ["Cat", "Centaur", "David", "Dog", "Gorilla", "Horse", "Michael", "Victoria", "Wolf"]:
    try:
        ds.TOSCA(root='/tmp/TOSCA/', categories=[category])
    except Exception as e:
        print(f"TOSCA ... missing {name}", e)
    pass

for category in ["NoNoise", "Noisy", "VarDensity", "NoisyAndVarDensity"]:
    try:
        ds.PCPNetDataset(root='/tmp/PCPNetDataset/', category=category)
    except Exception as e:
        print(f"PCPNetDataset ... missing {category}", e)
    pass

ds.S3DIS(root='/tmp/S3DIS/')
ds.GeometricShapes(root='/tmp/GeometricShapes/')
ds.BitcoinOTC(root='/tmp/BitcoinOTC/')
ds.ICEWS18(root='/tmp/ICEWS18/')
ds.GDELT(root='/tmp/GDELT/')

for pair in ["en_zh", "en_fr", "en_ja", "zh_en", "fr_en", "ja_en"]:
    try:
        ds.DBP15K(root='/tmp/DBP15K/', pair=pair)
    except Exception as e:
        print(f"DBP15K ... missing {pair}", e)
    pass

for category in ["Car", "Duck", "Face", "Motorbike", "Winebottle"]:
    try:
        ds.WILLOWObjectClass(root='/tmp/WILLOWObjectClass/', category=category)
    except Exception as e:
        print(f"WILLOWObjectClass ... missing {category}", e)
    pass

for category in ["Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Diningtable", "Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep", "Sofa", "Train", "TVMonitor"]:
    try:
        ds.PascalVOCKeypoints(root='/tmp/PascalVOCKeypoints/', category=category)
    except Exception as e:
        print(f"PascalVOCKeypoints ... missing {category}", e)

for category in ["Aeroplane", "Bicycle", "Bird", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Diningtable", "Dog", "Horse", "Motorbike", "Person", "Pottedplant", "Sheep", "Sofa", "Train", "TVMonitor"]:
    try:
        ds.PascalPF(root='/tmp/PascalPF/', category=category)
    except Exception as e:
        print(f"PascalPF ... missing {category}", e)
    pass

# ds.SNAPDataset(root='/tmp/SNAPDataset/')
# ds.SuiteSparseMatrixCollection(root='/tmp/SuiteSparseMatrixCollection/')
# ds.TrackMLParticleTrackingDataset(root='/tmp/TrackMLParticleTrackingDataset/')
ds.AMiner(root='/tmp/AMiner/')
ds.WordNet18(root='/tmp/WordNet18/')
ds.WordNet18RR(root='/tmp/WordNet18RR/')
ds.WikiCS(root='/tmp/WikiCS/')

for name in ["Cornell", "Texas", "Wisconsin"]:
    try:
        ds.WebKB(root='/tmp/WebKB/', name=name)
    except Exception as e:
        print(f"WebKB ... missing {name}", e)
    pass

for name in ["chameleon", "crocodile", "squirrel"]:
    try:
        ds.WikipediaNetwork(root='/tmp/WikipediaNetwork/', name=name)
    except Exception as e:
        print(f"WikipediaNetwork ... missing {name}", e)
    pass

ds.Actor(root='/tmp/Actor/')
ds.OGB_MAG(root='/tmp/OGB_MAG/')
ds.DBLP(root='/tmp/DBLP/')
ds.MovieLens(root='/tmp/MovieLens/')
ds.IMDB(root='/tmp/IMDB/')
ds.LastFM(root='/tmp/LastFM/')

for name in ["ACM", "DBLP", "Freebase", "IMDB"]:
    try:
        ds.HGBDataset(root='/tmp/HGBDataset/', name=name)
    except Exception as e:
        print(f"HGBDataset ... missing {name}", e)
    pass

# ds.JODIEDataset(root='/tmp/JODIEDataset/')

for homophily in [0.1,0.3,0.5,0.7,0.9]:
    ds.MixHopSyntheticDataset(root='/tmp/MixHopSyntheticDataset/', homophily=homophily)

for name in ["politifact", "gossipcop"]:
    for feature in ["profile", "spacy", "bert", "content"]:
        try:
            ds.UPFD(root='/tmp/UPFD/', name=name, feature=feature)
        except Exception as e:
            print(f"UPFD ... missing {name},{feature}", e)
        pass

ds.GitHub(root='/tmp/GitHub/')
ds.FacebookPagePage(root='/tmp/FacebookPagePage/')
ds.LastFMAsia(root='/tmp/LastFMAsia/')
ds.DeezerEurope(root='/tmp/DeezerEurope/')

for name in ["HU", "HR", "RO"]:
    try:
        ds.GemsecDeezer(root='/tmp/GemsecDeezer/', name=name)
    except Exception as e:
        print(f"GemsecDeezer ... missing {name}", e)
    pass

for name in ["DE", "EN", "ES", "FR", "PT", "RU"]:
    try:
        ds.Twitch(root='/tmp/Twitch/', name=name)
    except Exception as e:
        print(f"Twitch ... missing {name}", e)
    pass

for name in ["USA", "Brazil", "Europe"]:
    try:
        ds.Airports(root='/tmp/Airports/', name=name)
    except Exception as e:
        print(f"Airports ... missing {name}", e)
    pass

ds.BAShapes()
ds.MalNetTiny(root='/tmp/MalNetTiny/')
ds.OMDB(root='/tmp/OMDB/')
ds.PolBlogs(root='/tmp/PolBlogs/')
ds.EmailEUCore(root='/tmp/EmailEUCore/')
# ds.StochasticBlockModelDataset(root='/tmp/StochasticBlockModelDataset/')
# ds.RandomPartitionGraphDataset(root='/tmp/RandomPartitionGraphDataset/')

for name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
    try:
        ds.LINKXDataset(root='/tmp/LINKXDataset/', name=name)
    except Exception as e:
        print(f"LINKXDataset ... missing {name}", e)
    pass

ds.EllipticBitcoinDataset(root='/tmp/EllipticBitcoinDataset/')
# ds.DGraphFin(root='/tmp/DGraphFin/')
