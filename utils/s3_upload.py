import io
import boto3

def upload_dataframe_to_s3(df, bucket, key, sep=';'):
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, sep=sep)
        csv_buffer.seek(0)

        s3 = boto3.client('s3')
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"Plik pomyślnie zapisany w S3: s3://{bucket}/{key}")
        return True
    except Exception as e:
        print(f"Błąd podczas wysyłania do S3: {e}")
        return False