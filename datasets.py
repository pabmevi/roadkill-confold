from foldrm import Classifier
import numpy as np

import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

from ModifiedClass import MyClassifier
import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier
        
def final_extinctionrisk(data_path='datasets/Extinction/traits_finalnoNA_26Nov25.csv'):
    attrs = [#"Residential_IUCN",
             "Agriculture_IUCN",#"Energy_IUCN",
             "Transportation_IUCN",
             "Biological_Use_IUCN",#"Human_Intrusions_IUCN",
             #"Natural_Modifications_IUCN",
             "Invasive_IUCN",#"Pollution_IUCN",
             #"Geological_IUCN",
             "Climate_IUCN",#"n_threats_IUCN","Habitat.Density_AVONET","Primary.Lifestyle_AVONET",
             "Beak.Length_Culmen_AVONET","Beak.Length_Nares_AVONET","Beak.Width_AVONET","Beak.Depth_AVONET",
             "Tarsus.Length_AVONET","Wing.Length_AVONET","Kipps.Distance_AVONET","Secondary1_AVONET","Hand.Wing.Index_AVONET",
             "Tail.Length_AVONET","Min.Latitude_AVONET","Max.Latitude_AVONET",#"ISL_BirdBase",
             "RLM_BirdBase",#"LAT_BirdBase",
             "NormMin_BirdBase","Elevational.Range_BirdBase","NormMax_BirdBase","HB_BirdBase","DB_BirdBase",
             #"Flightlessness_BirdBase",
             "Adult.survival_GenLength","Age.at.first.breeding_GenLength",
             "Maximum.longevity_GenLength","GenLength_GenLength","Foraging_BirdBehav","MatingSystem_BirdBehav",
             "NestPlacement_BirdBehav","Territoriality_BirdBehav","IslandDwelling_BirdBehav","LogNightLights_BirdBehav",
             "LogHumanPopulationDensity_BirdBehav",#"Order.IUCN","Family.IUCN",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak.Length_Culmen_AVONET","Beak.Length_Nares_AVONET","Beak.Width_AVONET","Beak.Depth_AVONET",
             "Tarsus.Length_AVONET","Wing.Length_AVONET","Kipps.Distance_AVONET","Secondary1_AVONET","Hand.Wing.Index_AVONET",
             "Tail.Length_AVONET","Min.Latitude_AVONET","Max.Latitude_AVONET","NormMin_BirdBase","Elevational.Range_BirdBase",
             "NormMax_BirdBase","Adult.survival_GenLength","Age.at.first.breeding_GenLength","Maximum.longevity_GenLength",
             "GenLength_GenLength", "LogNightLights_BirdBehav","LogHumanPopulationDensity_BirdBehav","Range_size",
             "Body_mass","Clutch_size"]
    label = "extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def extinction_birds(data_path='datasets/Extinction/AvoIUCNbehavMig.csv'):
    attrs = ['IslandEndemic','Mass','HWI','Habitat.x','Beak.Length.culmen','Beak.Length.nares',
            'Beak.Width','Beak.Depth','Tarsus.Length','Wing.Length','Kipps.Distance','Secondary1',
            'Tail.Length','LogRangeSize','Diet','Foraging','Migration','MatingSystem','NestPlacement','Territoriality',
            'IslandDwelling','LogClutchSize','LogNightLights','LogHumanPopulationDensity',
            'Extinct_full','Extinct_partial','Marine_full','Marine_partial','Migr_dir_full','Migr_dir_partial',
            'Migr_dir_local','Migr_disp_full','Migr_disp_partial','Migr_disp_local','Migr_altitudinal',
            'Irruptive','Nomad_full','Nomad_partial','Nomad_local','Resid_full','Resid_partial',
            'Unknown','Uncertain','Migratory_status','Migratory_status_2','Migratory_status_3']
    
    nums = ['Mass', 'HWI','Beak.Length.culmen','Beak.Length.nares','Beak.Width','Beak.Depth','Tarsus.Length',
            'Wing.Length','Kipps.Distance','Secondary1','Tail.Length','RedlistCategory','LogRangeSize',
            'LogBodyMass','LogClutchSize','LogNightLights','LogHumanPopulationDensity']
    label = 'Threat'
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    print('\n% extinction birds dataset loaded', np.shape(data))
    return model, data

def new_extinction_birds_nomissingvalues(data_path='datasets/Extinction/AvoBirdbBehGLength_noNA.csv'):
    attrs = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth","Tarsus.Length",
             "Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index","Tail.Length",
             "Habitat.Density","Primary.Lifestyle","Min.Latitude","Max.Latitude","RLM","LAT",
             "MinAltitude","Elevational.Range","MaxAltitude","HB","DB","Flightlessness","Order",
             "Habitat","Foraging","MatingSystem","NestPlacement","Territoriality","IslandDwelling",
             "LogNightLights","LogHumanPopulationDensity","Family","Range_size",
             "Body_mass","Clutch_size","Diet","Migration","IslandEndemic","Adult.survival",
             "Age.at.first.breeding","Maximum.longevity","GenLength","extinction_risk"
]
    nums = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth","Tarsus.Length",
            "Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index","Tail.Length", "Min.Latitude",
            "Max.Latitude", "MinAltitude","Elevational.Range","MaxAltitude", "HB", "DB","LogNightLights",
            "LogHumanPopulationDensity", "Range_size", "Body_mass", "Clutch_size"]
    label = "extinction_risk"
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    print('\n% extinction birds dataset loaded', np.shape(data))
    return model, data

def new_extinction_birds_imputed(data_path='datasets/Extinction/BirdTraits_15imp_6Nov.csv'):
    attrs = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth","Tarsus.Length",
             "Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index","Tail.Length",
             "Habitat.Density","Primary.Lifestyle","Min.Latitude","Max.Latitude","RLM","LAT",
             "MinAltitude","Elevational.Range","MaxAltitude","HB","DB","Flightlessness","Order",
             "Habitat","Foraging","MatingSystem","NestPlacement","Territoriality","IslandDwelling",
             "LogNightLights","LogHumanPopulationDensity","Family","Range_size",
             "Body_mass","Clutch_size","Diet","Migration","IslandEndemic","Adult.survival",
             "Age.at.first.breeding","Maximum.longevity","GenLength","extinction_risk"
]
    nums = ["Beak.Length_Culmen","Beak.Length_Nares","Beak.Width","Beak.Depth","Tarsus.Length",
            "Wing.Length","Kipps.Distance","Secondary1","Hand.Wing.Index","Tail.Length", "Min.Latitude",
            "Max.Latitude", "MinAltitude","Elevational.Range","MaxAltitude", "HB", "DB","LogNightLights",
            "LogHumanPopulationDensity", "Range_size", "Body_mass", "Clutch_size"]
    label = "extinction_risk"
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path) # Use the argument here
    print('\n% extinction birds dataset loaded', np.shape(data))
    return model, data


