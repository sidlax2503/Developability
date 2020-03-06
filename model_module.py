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
        self.sample_fraction=sample_fraction
        self.model_name=model_in+'_'+model_out+'_'+model_architecture+'_'+str(sample_fraction)
        self.trials_file='./trials/'+self.model_name+'.pkl'
        self.stats_file='./model_stats/'+self.model_name+'.pkl'
        self.plotpairs_file='./plotpairs/'+self.model_name+'.pkl'
        self.figure_file='./figures/'+self.model_name+'.png'


        self.build_architecture(model_architecture) #ridge,forest,svm,nn(emb,)
        self.load_hyp()
        self.load_model_stats()
        self.plotpairs_cv=[[],[],[]] #1st repeat CV (concat all splits) predictions [true y, pred y, cat var]
        self.plotpairs_test=[[],[],[]] #1st repeat test predictions 
        self.load_plotpairs()

        self.plot_type=None

    def parent_warning(self):
        print('im in parent class')

    def save_plotpairs(self):
        with open (self.plotpairs_file,'wb') as f:
            pickle.dump([self.plotpairs_cv,self.plotpairs_test],f)

    def load_plotpairs(self):
        try:
            [self.plotpairs_cv,self.plotpairs_test]=pickle.load(open(self.plotpairs_file,'rb'))
        except:
            print('No plot data available')
            [self.model_plotpairs_cv,self.model_plotpairs_test]=[[[],[],[]],[[],[],[]]]

    def save_model_stats(self):
        with open (self.stats_file,'wb') as f:
            pickle.dump(self.model_stats,f)

    def load_model_stats(self):
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

    def build_architecture(self, model_architecture):
        'load architecture class which sets hyp space'
        self._model=get_model(model_architecture)

    def evaluate_model_common(self):
        true_pred_pairs=[]
        for i in self.data_pairs:
            train_x,train_y,train_cat_var=self.format_modelIO(i[0])
            test_x,test_y,test_cat_var=self.format_modelIO(i[1])
            self._model.model.fit(train_x,train_y)
            test_prediction=self._model.model.predict(test_x).squeeze().tolist()
            true_pred_pairs.append([test_y,test_prediction,test_cat_var])
        return true_pred_pairs

    def evaluate_model_cv(self):
        'train the repeated kfold dataset. Caclulate average across splits of one dataset, then average across repeats of dataset'
        true_pred_pairs=self.evaluate_model_common()
        cv_mse=[] #average mse values for each repeat of the spliting
        for i in range(0,len(true_pred_pairs),self.num_cv_splits):
            split_mse=[] #mse values for each split of the data
            for j in range(i,i+self.num_cv_splits):
                split_mse.append(mse(true_pred_pairs[j][0],true_pred_pairs[j][1]))
            cv_mse.append(np.average(split_mse))
        cur_cv_mse=np.average(cv_mse)
        if cur_cv_mse < self.model_stats['cv_avg_loss']:
            self.model_stats['cv_avg_loss']=cur_cv_mse
            self.model_stats['cv_std_loss']=np.std(cv_mse)

            self.model_plotpairs_cv=[[],[],[]] #if best CV loss, save the predictions for the first repeat across the splits 
            for i in range(0,self.num_cv_splits):
                self.plotpairs_cv[0]=self.plotpairs_cv[0]+true_pred_pairs[i][0]
                self.plotpairs_cv[1]=self.plotpairs_cv[1]+true_pred_pairs[i][1]
                self.plotpairs_cv[2]=self.plotpairs_cv[2]+true_pred_pairs[i][2]
        return cur_cv_mse

    def evaluate_model_test(self):
        'train the reapeated training data. Calculate average loss on test set to average out model randomness'
        true_pred_pairs=self.evaluate_model_common()
        mse_list=[]
        for i in true_pred_pairs:
            mse_list.append(mse(i[0],i[1]))
        cur_test_mse=np.average(mse_list)
        self.model_stats['test_avg_loss']=cur_test_mse
        self.model_stats['test_std_loss']=np.std(mse_list)
        self.model_plotpairs_test=[[],[],[]]
        self.plotpairs_test[0]=self.plotpairs_test[0]+true_pred_pairs[0][0]
        self.plotpairs_test[1]=self.plotpairs_test[1]+true_pred_pairs[0][1]
        self.plotpairs_test[2]=self.plotpairs_test[2]+true_pred_pairs[0][2]
        return cur_test_mse

    # def predict_model(self):
    #     'useing trained model, predict test set'
    #     self.parent_warning()

    # def save_predictions(self):
    #     'updata dataframe with new column of predicted values from model'
    #     self.parent_warning()

    def print_tpe_trials(self):
        print(pd.DataFrame(list(self.tpe_trials.results)))

    def get_best_trial(self):
        'sort trials by loss, return best trial'
        if len(self.tpe_trials)>0:
            if len(self.tpe_trials)<self.num_hyp_trials:
                print('Warning: Not fully tested hyperparameters: ' + str(len(self.tpe_trials)) + '<' + str(self.num_hyp_trials))
            sorted_trials = sorted(self.tpe_trials.results, key=lambda x: x['loss'], reverse=False)
            return sorted_trials[0]
        print('no trials found')

    def load_hyp(self):
        'load hyperopt trials'
        try:  # try to load an already saved trials object
            self.tpe_trials = pickle.load(open(self.trials_file, "rb"))
        except:
            self.tpe_trials = Trials()

    def save_hyp(self):
        'save hyperopt trials, refresh best trial'
        with open(self.trials_file, "wb") as f:
            pickle.dump(self.tpe_trials, f)

    # def save_model(self):
    #     'save the trained model'
    #     self.parent_warning()

    def format_modelIO(self,df):
        'based upon model architecture and catagorical variables create the numpy input (x) and output (y) for the model'
        df_local,cat_var,y=self.get_output_and_explode(df) #set y, do output firest to explode cat variables
        x_a=self.get_input_seq(df_local) #set xa (OH seq, Ord seq, assay, control)
        x=load_format_data.mix_with_cat_var(x_a,cat_var) #mix xa with cat variables
        return x,y,cat_var

    def make_cv_dataset(self):
        'create list of subtraining/validation by repeated cv of training data'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        kf=RepeatedKFold(n_splits=self.num_cv_splits,n_repeats=self.num_cv_repeats)
        train,validate=[],[]
        for train_index, test_index in kf.split(np.zeros(len(local_df))):
            train.append(local_df.iloc[train_index])
            validate.append(local_df.iloc[test_index])
        self.data_pairs=zip(train,validate)

    def make_test_dataset(self,num_test_repeats):
        'create list of full training set/test set for repeated model performance evaluation'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        train,test=[],[]
        for i in range(num_test_repeats):
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
            self.make_test_dataset(10)

    def hyperopt_obj(self,space):
        'for a given hyperparameter set, build model arch, evaluate model, return validation loss'
        self.set_model_state(cv=True)
        self._model.set_model(space)
        loss=self.evaluate_model()

        return {'loss': loss, 'status': STATUS_OK ,'hyperparam':space}

    def cross_validate_model(self):
        'use hpyeropt to determine hyperparameters for self.tpe_trials'
        if len(self.tpe_trials)<self.num_hyp_trials:
            tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=len(self.tpe_trials)+10)
            self.save_hyp()
            self.save_model_stats()
            self.save_plotpairs()
        else:
            print('Already done with cross-validation')

    def test_model(self):
        'using the best hyperparameters, train using full training dataset and predict test set'
        self.set_model_state(cv=False)
        self._model.set_model(self.get_best_trial()['hyperparam'])
        loss=self.evaluate_model()
        self.save_model_stats()
        self.save_plotpairs()
        print('test loss=',str(loss))

    def plot(self):
        figure=self.plot_type(self)
        figure.fig.savefig(self.figure_file)
        figure.fig.clf()
