import boto3
import pandas as pd
import warnings
from io import BytesIO
warnings.filterwarnings("ignore")

class S3DataProcessor:
    def __init__(self, aws_access_key_id, aws_secret_access_key, bucket_name, directory_names):
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.bucket_name = bucket_name
        self.directory_names = directory_names
        self.s3_client = self._create_s3_client()

    def _create_s3_client(self):
        """ function to create s3 client """
        return boto3.client(
            's3',
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

    def _read_parquet_from_s3(self, file_key):
        """ funciton to read parquet data from s3 """
        s3_object = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
        parquet_data = s3_object['Body'].read()
        return pd.read_parquet(BytesIO(parquet_data))

    def _get_parquet_files(self, directory_name):
        response = self.s3_client.list_objects(Bucket=self.bucket_name, Prefix=directory_name)
        return [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]

    def process_data(self):
        """ function to process data and load into dataframes"""
        dfs_social = []
        dfs_news = []
        dfs_blog = []
        for directory_name in self.directory_names:
            parquet_files = self._get_parquet_files(directory_name)
            for file in parquet_files:
                df = self._read_parquet_from_s3(file)
                if directory_name == 'data-science/social':
                    dfs_social.append(df)
                elif directory_name == 'data-science/news':
                    dfs_news.append(df)
                elif directory_name == 'data-science/blog':
                    dfs_blog.append(df)
        return dfs_social, dfs_news, dfs_blog