# Databricks notebook source
df = spark.createDataFrame(
    [
        {"message": "I love using Hugging Face!"},
        {"message": "I hate using Hugging Face!"}
    ],
    schema="message string"
)

# COMMAND ----------

from transformers import pipeline
from pyspark.sql.functions import pandas_udf, PandasUDFType

sentiment_analysis = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

@pandas_udf('string', PandasUDFType.SCALAR)
def sentiment_udf(v: pd.Series) -> pd.Series:
    return v.apply(lambda x: sentiment_analysis(x)[0]['label'])

df = df.withColumn('sentiment', sentiment_udf(df.message))

display(df)

# COMMAND ----------


