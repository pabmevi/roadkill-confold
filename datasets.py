import sys, os
sys.path.append(os.path.join(os.getcwd(), 'CONFOLD'))

import numpy as np
import pandas as pd
from CONFOLD.foldrm import Classifier

class MyClassifier(Classifier):
        def load_data(self, file, amount=-1):
            data, self.attrs = use_dataframe(file, self.attrs, self.label, self.numeric, amount)
            return data
        

def use_dataframe(file, attrs, label, numeric, amount):
    df = pd.read_csv(file, sep=',', on_bad_lines='skip') #get the dataframe

    #split into x and y
    df_x = df[attrs]
    df_y = df[label]
    result = pd.concat([df_x,df_y], axis=1)
    attrs.append(label)
    return result,attrs
        
def final_extinctionrisk(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration","Extinction_risk"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_noth(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration","Extinction_risk"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_dataframe(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def final_extinctionrisk_noth_dataframe(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data


def rk_mammals(data_path='datasets/Extinction/RkTraits_CONFOLD.csv'):
    attrs = ["decimalLatitude","decimalLongitude","Order","adult_mass_g","max_longevity_d",
             "age_first_reproduction_d","litter_size_n","litters_per_year_n","dispersal_km",
             "density_n_km2","home_range_km2","dphy_invertebrate","dphy_plant","det_vend",
             "det_vect","det_scav","det_diet_breadth_n","altitude_breadth_m","EQ",
             "habitat_Forest","habitat_Savanna","habitat_Shrubland","habitat_Grassland",
             "habitat_Wetlands","habitat_Desert","habitat_Artificial_Terrestrial",
             "n_habitats","human_footprint","roadkill_category"]

    nums = ["decimalLatitude","decimalLongitude","adult_mass_g","max_longevity_d",
             "age_first_reproduction_d","litter_size_n","litters_per_year_n","dispersal_km",
             "density_n_km2","home_range_km2","dphy_invertebrate","dphy_plant","det_vend",
             "det_vect","det_scav","det_diet_breadth_n","altitude_breadth_m","EQ"]
    label = "roadkill_category"
    
    model = Classifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def extinction_birds2(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data

def extinction_birds2noth(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
    label = "Extinction_risk"
    
    model = MyClassifier(attrs=attrs, numeric=nums, label=label)
    data = model.load_data(data_path)
    return model, data


def final_extinctionrisk(data_path='datasets/Extinction/traits_combined_noNA_5Dece25.csv'):
    attrs = ["Order","Family","Agriculture","Hunting","Invasive_species","Climate_change",
             "Beak_length_culmen","Beak_depth",
             "Tarsus_length","Wing_length","Hand_wing_index","Tail_length","Minimum_latitude","Maximum_latitude",
             "Primary_lifestyle","Island_restricted_breeding","Latitudinal_range","Elevational_range","Habitat_breadth",
             "Diet_breadth","Realm","Minimum_elevation","Maximum_elevation","Adult_survival_annual","Generation_length",
             "Range_size","Body_mass","Clutch_size","Diet","Habitat","Migration"]
    
    nums = ["Beak_length_culmen","Beak_depth","Tarsus_length","Wing_length","Hand_wing_index","Tail_length",
            "Minimum_latitude","Maximum_latitude","Minimum_elevation","Elevational_range","Maximum_elevation",
            "Habitat_breadth","Diet_breadth","Adult_survival_annual","Generation_length","Range_size","Body_mass",
            "Clutch_size"]
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


