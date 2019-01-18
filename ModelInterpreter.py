import shap
import eli5
from IPython.display import display
from pdpbox import info_plots, pdp
import matplotlib.pyplot as plt

class ModelInterpreter():
    def __init__(self, model, feature_names, X_train, sample_size):
        self.model = model
        self.feature_names = feature_names
        self.X_train = X_train
        self.sample_size = sample_size
    
    def plot_weights(self,num_weights=None):
        display(eli5.show_weights(self.model,top=num_weights,feature_names=self.feature_names))
    
    def show_summary(self, model_type=None):
        # load JS visualization code to notebook
        shap.initjs()
        self.explainer = None
        if model_type == 'linear':
            self.explainer = shap.LinearExplainer(self.model,self.X_train.sample(self.sample_size,random_state=42))
        elif model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model,self.X_train.sample(self.sample_size,random_state=42))
        else:
            self.explainer = shap.KernelExplainer(self.model,self.X_train.sample(self.sample_size,random_state=42))
        shap_values = self.explainer.shap_values(self.X_train.sample(self.sample_size,random_state=42))
        display(shap.summary_plot(shap_values,self.X_train.sample(self.sample_size,random_state=42)))
        display(shap.summary_plot(shap_values,self.X_train.sample(self.sample_size,random_state=42),plot_type='bar'))

    def show_SHAP_PDP(self, features=[]):
        for f in features:
            if len(f) > 1:
                continue
            shap.dependence_plot(f, 
                                 self.explainer.shap_values(self.X_train.sample(self.sample_size,random_state=42)), 
                                 self.X_train.sample(self.sample_size,random_state=42))
            
    def show_SHAP_PDP_interaction(self, features=[]):
        for f in features:
            try:
                shap.dependence_plot(tuple(f), 
                                 self.explainer.shap_interaction_values(self.X_train.sample(self.sample_size,random_state=42)), 
                                 self.X_train.sample(self.sample_size,random_state=42))
            except:
                print("Linear estimators don't have interaction values.")
                return
            
    def show_ICE_actual(self, features=[], feature_names=[]):
        if len(features) != len(feature_names):
            print("features and feature names must have same size")
            return
        for f,n in zip(features,feature_names):
            info_plots.actual_plot(self.model,self.X_train.sample(self.sample_size,random_state=42),
                      feature=f,feature_name=n,predict_kwds={})
            plt.xticks(rotation=90)
            plt.show()
    
    def show_PDP_isolate(self, features=[]):
        for f in features:
            pdp_isolate = pdp.pdp_isolate(self.model,self.X_train.sample(self.sample_size,random_state=42),
                      model_features=self.feature_names,feature=f,predict_kwds={})
            pdp.pdp_plot(pdp_isolate,feature_name=f)
            plt.xticks(rotation=90)
            plt.show()
    
    def show_PDP_interact(self, features=[]):
        for f in features:
            pdp_interact = pdp.pdp_interact(self.model,self.X_train.sample(self.sample_size,random_state=42),
                                   model_features=self.feature_names,
                                            features=f)
            pdp.pdp_interact_plot(pdp_interact,feature_names=f)
            plt.xticks(rotation=90)
            plt.show()
            
    def show_all(self,model_type,features,feature_names,interact_feat):
        self.plot_weights()
        self.show_summary(model_type)
        self.show_SHAP_PDP(features)
        self.show_SHAP_PDP_interaction(interact_feat)
        self.show_ICE_actual(features,feature_names)
        self.show_PDP_isolate(features)
        self.show_PDP_interact(interact_feat)
