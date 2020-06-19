import numpy as np
import pandas as pd
import pickle
from hyperopt import Trials,fmin,tpe,STATUS_OK
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
import load_format_data
from model_architectures import get_model
from plot_model import model_plot

class model:
    '''The model class will cross-validate the training set to determine hyperparameters
    then use the set hyperparameters to evaluate against a test set and save predictions''' 

    def __init__(self, model_in, model_out, model_architecture, sample_fraction):
        ## This creates an instance of an object of type model 
        ## Of the given inputs model_in, model_out and model_architecture are strings
        ## sample fractions is a float of value between 0.0-1.0
        self.sample_fraction=sample_fraction
        
        self.model_architecture=model_architecture
        ## Notice how the class variables sample_fraction and model_architecture
        ## are attributed to their respective inputs
        
        self.model_name=model_in+'_'+model_out+'_'+model_architecture+'_'+str(sample_fraction)
        ## model_name is a string combination of all the inputs. 
        
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        ## The above class variables are string pathways to respective pickle files needed to construct the model
        ## These class variables are also dependent on the previously mentioned variable model_name
        
        self.figure_file='./figures/'+self.model_name+'.png'
        ## figure_file is a string pathway to access the corresponding model image given the model_name
        
        self.model_loc='./models/'+self.model_name
        ## model_loc is a string pathway showing the location of the model.
        
        
        self._model = get_model(model_architecture) #ridge,forest,svm,nn(emb,)
        ## This sets a self._model to the regression typoe used to build the model, model_architercture is a string input
        
        ## If-else statement checks if the following model has already been run. 
        ## if it does exsist then it access them and assigns them to respective class variable
        ## else it sets them to a default value. 
        if(self.load_hyp()==True):
            self.model_stats = pickle.load(open(self.stats_file, "rb"))
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
            
        else:
            print('No previous trial data, model data or plot data available')
            self.model_stats= {'cv_avg_loss': np.inf,'cv_std_loss': [],'test_avg_loss': np.inf,'test_std_loss': []}
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]
        
        self.plot_type=None

    def parent_warning(self):
        print('im in parent class')

### Question: should we be amending the model_architecture and sample fraction
    def update_model_name(self,model_name):
        ## This is a setter function used to change the model_name, thereby the assay, model_architecture and corresponding
        ## trial data, model data and plot data. The function input is a string
        self.model_name=model_name
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        self.figure_file='./figures/'+self.model_name+'.png'
        self.model_loc='./models/'+self.model_name
        ## The above statements set the class variable established in the instantiation function to the new value
        self.load_hyp()
        self.load_model_stats()
        self.load_plotpairs()
        ## The above functions check if trial data, model data and plot data already exsist for the given model_name
        ## If it doesn exsist then it sets it respectively or else it sets it to a 

    def save_plotpairs(self):
        ## This function navigates to the plotpairs directory and saves 
        ## the plotpairs_cv and plotpairs_test data as a pickle file under the name of the model_name
        with open (self.plotpairs_file,'wb') as f:
            pickle.dump([self.plotpairs_cv,self.plotpairs_test],f)

    def load_plotpairs(self):
        ## This function checks whether plot data is available in the plotpairs directory
        ## if it is available then it sets the ploatpairs_cv and plotpairs_test to the data else it defualts them
        try:
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
        except:
            print('No plot data available')
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]

    def save_model_stats(self):
        ## This function navigates to the model_stats directory and saves the 
        ## self.model_stats data as a pickle file under the model_name
        with open (self.stats_file,'wb') as f:
            pickle.dump(self.model_stats,f)

    def load_model_stats(self):
        ## This function checks whether model data is available in the model_stats directory
        ## if it is available then it sets the model_stats variable to the data else it defaults the variable
        try: 
            self.model_stats = pickle.load(open(self.stats_file, "rb"))
        except:
            print('No previous model data saved')
            self.model_stats= {
            'cv_avg_loss': np.inf,
            'cv_std_loss': [],
            'test_avg_loss': np.inf,
            'test_std_loss': []
            }
            print(self.model_name)

    def build_architecture(self, model_architecture):
        ## This function is used to create a model_architecure model object and make it a protected class variable
        ## also acts aas a setter function
        'load architecture class which sets hyp space'
        self._model=get_model(model_architecture)

    def load_hyp(self):
        ## This function checks whether trial data is available in the trials directory
        ## if it is available then it sets the tpe_trails variable to the data else it defaults the variable
        'load hyperopt trials'
        try:  # try to load an already saved trials object
            self.tpe_trials = pickle.load(open(self.trials_file, "rb"))
            return True
        except:
            self.tpe_trials = Trials()
            return False

    def save_hyp(self):
        ## This function navigates to the trials directory and saves the
        ## self.tpe_trials  data as a pickle file under the model name
        'save hyperopt trials, refresh best trial'
        with open(self.trials_file, "wb") as f:
            pickle.dump(self.tpe_trials, f)
    
    def format_modelIO(self,df):
        ## df is a dataframe object
        'based upon model architecture and catagorical variables create the numpy input (x) and output (y) for the model'
        ## This function only works for the objects x_to_assay_model or the x_to_yield_model defined in the submodels_module.py script
        ## If it is the x_to_assay_model then the .get_output_and_explode function calls the explode_assay function of the 
        ## load_format_data.py script with the assays specified in the instatiation of the object
        ## Else it is the x_to_yield_model then the .get_output_and_explode function calls thee explode_yield function of the load_format_data py script
        df_local,cat_var,y=self.get_output_and_explode(df) #set y, do output firest to explode cat variables
        ## Refer to the load_format_data functions of explode_yield and explode_assay to determine the value in df_local, cat_var and y
        x_a=self.get_input_seq(df_local) #set xa (OH seq, Ord seq, assay, control)
        x=load_format_data.mix_with_cat_var(x_a,cat_var) #mix xa with cat variables
        return x,y,cat_var
            
    def evaluate_model_common(self,space,save_model):
        ## The space input is a dictionary with the details regarding the parameter space
        true_pred_pairs=[]
        model_no=0
        for i in self.data_pairs:
            train_x,train_y,train_cat_var=self.format_modelIO(i[0])
            test_x,test_y,test_cat_var=self.format_modelIO(i[1])
            self._model.set_model(space,xa_len=len(train_x[0])-len(train_cat_var[0]), cat_var_len=len(train_cat_var[0]), lin_or_sig=self.lin_or_sig)
            self._model.fit(train_x,train_y)
            if save_model:
                self.save_model(model_no)
            test_prediction=self._model.model.predict(test_x).squeeze().tolist()
            true_pred_pairs.append([test_y,test_prediction,test_cat_var])
            model_no=model_no+1
        return true_pred_pairs

    def evaluate_model_cv(self,space,force_saveplots=False):
        'train the repeated kfold dataset. Caclulate average across splits of one dataset, then average across repeats of dataset'
        true_pred_pairs=self.evaluate_model_common(space,False)
        cv_mse=[] #average mse values for each repeat of the spliting
        for i in range(0,len(true_pred_pairs),self.num_cv_splits):
            split_mse=[] #mse values for each split of the data
            for j in range(i,i+self.num_cv_splits):
                split_mse.append(mse(true_pred_pairs[j][0],true_pred_pairs[j][1]))
            cv_mse.append(np.average(split_mse))
        cur_cv_mse=np.average(cv_mse)
        if force_saveplots or (cur_cv_mse < self.model_stats['cv_avg_loss']):
            self.model_stats['cv_avg_loss']=cur_cv_mse
            self.model_stats['cv_std_loss']=np.std(cv_mse)

            self.plotpairs_cv=[[],[],[]] #if best CV loss, save the predictions for the first repeat across the splits 
            for i in range(0,self.num_cv_splits):
                self.plotpairs_cv[0]=self.plotpairs_cv[0]+true_pred_pairs[i][0]
                self.plotpairs_cv[1]=self.plotpairs_cv[1]+true_pred_pairs[i][1]
                self.plotpairs_cv[2]=self.plotpairs_cv[2]+true_pred_pairs[i][2]
        return cur_cv_mse

    def evaluate_model_test(self,space):
        'train the reapeated training data. Calculate average loss on test set to average out model randomness'
        true_pred_pairs=self.evaluate_model_common(space,True)
        mse_list=[]
        for i in true_pred_pairs:
            mse_list.append(mse(i[0],i[1]))
        cur_test_mse=np.average(mse_list)
        self.model_stats['test_avg_loss']=cur_test_mse
        self.model_stats['test_std_loss']=np.std(mse_list)
        self.plotpairs_test=[[],[],[]]
        self.plotpairs_test[0]=self.plotpairs_test[0]+true_pred_pairs[0][0]
        self.plotpairs_test[1]=self.plotpairs_test[1]+true_pred_pairs[0][1]
        self.plotpairs_test[2]=self.plotpairs_test[2]+true_pred_pairs[0][2]
        return cur_test_mse

    def print_tpe_trials(self):
        print(pd.DataFrame(list(self.tpe_trials.results)))

    def get_best_trial(self):
        'sort trials by loss, return best trial'
        if len(self.tpe_trials)>0:
            if len(self.tpe_trials)<self.num_hyp_trials:
                print('Warning: Not fully tested hyperparameters: ' + str(len(self.tpe_trials)) + '<' + str(self.num_hyp_trials)+':'+self.model_name)
            sorted_trials = sorted(self.tpe_trials.results, key=lambda x: x['loss'], reverse=False)
            return sorted_trials[0]
        print('no trials found')

    def save_model(self,model_no):
        'save the trained model'
        ## Checks if the regression model for the data is based on neural network
        ## If it is saves all layer weights as a tensor flow in the models directors under the model name
        ## Else it saves the model as a pickle file in models directory under the model name
        if 'nn' in self.model_architecture:
            self._model.model.save_weights(self.model_loc+'_'+str(model_no)+'/')
        else:
            with open(self.model_loc+'_'+str(model_no)+'.pkl', "wb") as f:
                pickle.dump(self._model.model, f)

    def load_model(self,model_no):
        ## Checkis if the regression model for the data is based on neural network 
        ## If it is, it loads all the layer weights based on the network to the self._model class variable
        ## Else it loads the class variable self._model as usual
        if 'nn' in self.model_architecture:
            self._model.model.load_weights(self.model_loc+'_'+str(model_no)+'/').expect_partial()
        else:
            self._model.model=pickle.load(open(self.model_loc+'_'+str(model_no)+'.pkl', "rb"))
#### Fix this below
    

    def make_cv_dataset(self):
        'create list of subtraining/validation by repeated cv of training data'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        kf=RepeatedKFold(n_splits=self.num_cv_splits,n_repeats=self.num_cv_repeats)
        train,validate=[],[]
        for train_index, test_index in kf.split(np.zeros(len(local_df))):
            train.append(local_df.iloc[train_index])
            validate.append(local_df.iloc[test_index])
        self.data_pairs=zip(train,validate)

    def make_test_dataset(self):
        'create list of full training set/test set for repeated model performance evaluation'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        train,test=[],[]
        for i in range(self.num_test_repeats):
            train.append(local_df)
            test.append(self.testing_df)
        self.data_pairs=zip(train,test)

    def set_model_state(self,cv):
        'create list of paired dataframes and determine how to calculate loss based upon cross-validaiton or applying to test set'
        if cv:
            self.evaluate_model=self.evaluate_model_cv
            self.make_cv_dataset() 
        else:
            self.evaluate_model=self.evaluate_model_test
            self.make_test_dataset()

    def hyperopt_obj(self,space):
        'for a given hyperparameter set, build model arch, evaluate model, return validation loss'
        self.set_model_state(cv=True)
        loss=self.evaluate_model(space)

        return {'loss': loss, 'status': STATUS_OK ,'hyperparam':space}

    def cross_validate_model(self):
        'use hpyeropt to determine hyperparameters for self.tpe_trials'
        if len(self.tpe_trials)<self.num_hyp_trials:
            if 'nn' in self.model_architecture:
                for i in range(10):
                    max_evals=min(len(self.tpe_trials)+5,self.num_hyp_trials)
                    tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=max_evals)
                    self.save_hyp()
                    self.save_model_stats()
                    self.save_plotpairs()
            else:
                tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=self.num_hyp_trials)
                self.save_hyp()
                self.save_model_stats()
                self.save_plotpairs()
        else:
            print('Already done with cross-validation')
            # self.set_model_state(cv=True)
            # self.evaluate_model(self.get_best_trial()['hyperparam'],force_saveplots=True)

    def test_model(self):
        'using the best hyperparameters, train using full training dataset and predict test set'
        self.set_model_state(cv=False)
        loss=self.evaluate_model(self.get_best_trial()['hyperparam'])
        self.save_model_stats()
        self.save_plotpairs()
        print('test loss=',str(loss))

    def plot(self):
        figure=self.plot_type(self)
        figure.fig.savefig(self.figure_file)
        figure.fig.clf()
