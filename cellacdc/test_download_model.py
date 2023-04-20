from cellacdc import myutils

model_name = 'deepsea' # 'segment_anything'
_, model_path = myutils.get_model_path(model_name, create_temp_dir=False)
model_exists = myutils.check_model_exists(model_path, model_name)

print(model_path)
print(model_exists)

myutils.download_model(model_name)