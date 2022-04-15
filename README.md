# SmartImputer
This function is a smart imputer, which gets X with missing data (np.nan) and uses RandomForestClassifier and RandomForestRegressor to predict all missing values.
    The passing through dic should be in this format:
    
    dic={"numerics":numerical_features,
     "categories":categorical_features,
     "binaries":binary_features,
     "passthrough":passthrough_features,
     "drops":drop_features
    }
