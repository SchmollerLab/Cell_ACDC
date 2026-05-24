import os

from cellacdc import myutils, load

myutils.check_install_yeaz()

custom_weights_json_filename = 'custom_weights_name_filepath.json'

def add_model_filepath(name: str, filepath: os.PathLike):
    _, model_folderpath = myutils.get_model_path(
        'YeaZ_v2', create_temp_dir=False
    )
    custom_weights_json_file = os.path.join(
        model_folderpath, custom_weights_json_filename
    )
    custom_weights_mapper = {}
    if os.path.exists(custom_weights_json_file):
        custom_weights_mapper = load.read_json(
            custom_weights_json_file, 
            desc='YeaZ_v2 custom weights filepath info'
        )
    
    custom_weights_mapper[name] = filepath
    load.write_json(custom_weights_mapper, custom_weights_json_file)
    
def load_models_filepath():
    values = [
        'Phase contrast',
        'Bright-field',
        'Fission yeast'
    ]
    mapper = {
        'Phase contrast': 'weights_budding_PhC_multilab_0_1',
        'Bright-field': 'weights_budding_BF_multilab_0_1',
        'Fission yeast': 'weights_fission_multilab_0_2'
    }
    _, model_folderpath = myutils.get_model_path(
        'YeaZ_v2', create_temp_dir=False
    )
    mapper = {
        name: os.path.join(model_folderpath, filename) 
        for name, filename in mapper.items()
    }
    
    custom_weights_json_file = os.path.join(
        model_folderpath, custom_weights_json_filename
    )
    if not os.path.exists(custom_weights_json_file):
        return values, mapper
    
    custom_weights_mapper = load.read_json(
        custom_weights_json_file, 
        desc='YeaZ_v2 custom weights filepath info'
    )
    values.extend(custom_weights_mapper.keys())
    mapper = {**mapper, **custom_weights_mapper}
    
    return values, mapper