from modelarts.session import Session
session = Session() 

bucket_path="coding 3/datasets/high resolution images/DIV2K_train_HR.zip" 
session.download_data(bucket_path=bucket_path, path="./DIV2K_train_HR.zip")

bucket_path="coding 3/datasets/high resolution images/DIV2K_valid_HR.zip" 
session.download_data(bucket_path=bucket_path, path="./DIV2K_valid_HR.zip")