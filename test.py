from src.DimondPricePrediction.pipelines.prediction_pipeline import CustomData



custdataobj = CustomData(0.3,62.1,58,4.27,4.29,2.66,"Ideal","E","SI1")

data = custdataobj.get_data_as_dataframe()

print(data)