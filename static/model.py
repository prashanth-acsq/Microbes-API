from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier 


class Model(object):
    def __init__(self, model_name: str, preprocessor, seed: int):

        self.model_name = model_name

        if self.model_name == "lgr":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", LogisticRegression(random_state=seed, max_iter=1000)),
                ]
            )
        
        elif self.model_name == "knc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", KNeighborsClassifier()),
                ]
            )

        
        elif self.model_name == "dtc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", DecisionTreeClassifier(random_state=seed)),
                ]
            )

        elif self.model_name == "etc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", ExtraTreeClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "rfc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", RandomForestClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "gbc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", GradientBoostingClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "abc":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", AdaBoostClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "etcs":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", ExtraTreesClassifier(random_state=seed)),
                ]
            )
        
        elif self.model_name == "gnb":
            self.model = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", GaussianNB()),
                ]
            )