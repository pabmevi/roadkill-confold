import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier
        
def final_extinctionrisk(data_path='datasets/Extinction/traits_finalnoNA_29Nov25.csv'):
    attrs = [#"Agriculture","Biological_use_hunting",
             #"Invasive_species","Climate_change",
             "Primary_lifestyle",
             "Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length",
             "Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Island_restricted_breeding","Realm","Latitudinal_range","Minimum_elevation",
             "Elevational_range","Maximum_elevation","Habitat_breadth","Diet_breadth",
             "Adult_survival_annual","Generation_length","Foraging_strategy",
             "Nest_placement","Territoriality","Night_lights","Human_population_density",
             "Order","Family","Range_size","Body_mass","Clutch_size","Diet","Habitat",
            "Migration","Extinction_risk"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length",
            "Wing_length","Hand_wing_index","Tail_length","Minimum_latitude",
            "Maximum_latitude","Minimum_elevation","Elevational_range",
            "Maximum_elevation","Habitat_breadth","Diet_breadth","Adult_survival_annual",
            "Generation_length","Night_lights","Human_population_density",
            "Range_size","Body_mass","Clutch_size"]
    label = "Extinction_risk"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
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


