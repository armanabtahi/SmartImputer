import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def SmartImputer(X,dic):

    #The passing through dic should be like this with list of different feature types
    #dic={"numerics":numerical_features,
    # "categories":categorical_features,
    # "binaries":binary_features,
    # "passthrough":passthrough_features,
    # "drops":drop_features
    #}

    numerical_features=dic["numerics"]
    categorical_features=dic["categories"]
    binary_features=dic["binaries"]
    passthrough_features=dic["passthrough"]
    drop_features=dic["drops"]

    Impute_list=X.columns[X.isna().any()].tolist()
    numeric_transformer=make_pipeline(SimpleImputer(strategy="median"),StandardScaler())
    categorical_transformer=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown='ignore',sparse=False,drop="first"))
    binary_transformer=make_pipeline(SimpleImputer(strategy="most_frequent"),OneHotEncoder(handle_unknown='ignore',drop="if_binary",sparse=False))

    preprocessor=make_column_transformer(
        (numeric_transformer,numerical_features),
        (categorical_transformer,categorical_features),
        (binary_transformer,binary_features),
        ("passthrough",passthrough_features),
        ("drop",drop_features),
    )

    list_features_type=["numerical_features","categorical_features","binary_features","passthrough_features"]
    pipe_rf_classifier = make_pipeline(preprocessor, RandomForestClassifier())
    pipe_rf_regressor = make_pipeline(preprocessor, RandomForestRegressor())

    for Impute_target in Impute_list:
        Impute_df_train=X[X[Impute_target].notna()]
        Impute_df_test=X[X[Impute_target].isna()]
        Impute_X_train=Impute_df_train.drop(columns=Impute_target)
        Impute_y_train=Impute_df_train[Impute_target]
        Impute_X_test=Impute_df_test.drop(columns=Impute_target)
        for types in list_features_type:
            if Impute_target in locals()[types]:
                locals()[types].remove(Impute_target)    
                if types=="numerical_features":
                    pipe=pipe_rf_regressor
                else:
                    pipe=pipe_rf_classifier  
                if types=="binary_features":
                    Impute_y_train=Impute_y_train.astype('int')
                pipe.fit(Impute_X_train,Impute_y_train);
                Impute_y_pred=pipe.predict(Impute_X_test)
                Impute_y_pred=pd.DataFrame(Impute_y_pred,index=Impute_X_test.index, columns=[Impute_target])
                #print(Impute_y_pred)
                X=pd.concat([pd.concat([Impute_X_train,Impute_y_train], axis=1),pd.concat([Impute_X_test,Impute_y_pred], axis=1)],axis=0).sort_index()
                locals()[types].append(Impute_target)
                break
    return X