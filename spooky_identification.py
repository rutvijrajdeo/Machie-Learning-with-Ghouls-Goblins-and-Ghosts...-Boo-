###bagging algorithm


import pandas as pd
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier,RandomForestClassifier,ExtraTreesClassifier,AdaBoostClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import time



DATA_PATH = "D:\PycharmProjects\ML Project\ghouls_goblins_ghosts\data\\train.csv"
test_DATA_PATH = "D:\PycharmProjects\ML Project\ghouls_goblins_ghosts\data\\test.csv"
dataset = pd.read_csv(DATA_PATH)
le=preprocessing.LabelEncoder()
color=le.fit(dataset['color'])
cl=le.transform(dataset['color'])
dataset.insert(5,"Ncolor",cl)


############for test#############
test_dataset = pd.read_csv(test_DATA_PATH)
test_color=le.fit(test_dataset['color'])
test_cl=le.transform(test_dataset['color'])
# print test_cl
test_dataset.insert(4,"Ncolor",test_cl)
test_array = test_dataset.values
# print test_array
# id = test_array[:,0]
# print "iddd",id,type(id)
test_id =test_dataset['id']


Xt = test_array[:,1:6]
# Y = test_array[:,7]

#####################################


array = dataset.values
# print array

X = array[:,1:6]
Y = array[:,7]
# print X
# print "Y"
# print Y
# Bagged Decision Trees for Classification
def bagginf_DecisionTree(X,Y):
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cart = DecisionTreeClassifier()
    num_trees = 100
    model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("DECISIONTREE ACC:    ",results.mean())

# Bagged Random FOrest for Classification
def RandomForestClass(X,Y):
    seed = 7
    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("RANDOMFOREST ACC:    ",results.mean())
    model.fit(X,Y)
    # print model.score(X,Y)
    result_rf = model.predict(Xt)
    # print result_rf

    rfc =RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)

    # param_grid = {
    #     'n_estimators':[200,700],
    #     'max_features':['auto','sqrt','log2']
    # }
    param_grid =  {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000],


    }
    start_time = time.time()
    print ("time started ",start_time)
    CV_rfc = GridSearchCV(estimator=rfc,param_grid=param_grid,cv=5)
    CV_rfc.fit(X, Y)
    print ("GRID SEARCH BEST PARAM: ",CV_rfc.best_params_)
    score = CV_rfc.score(X,Y)
    print("GRID SEARCH SCORE:   ", score)
    test_result = CV_rfc.predict(Xt)
    print ("TEST RESULT:    ",test_result)
    print ("time taken   ",time.time()-start_time)

def save_result(result):
    submission = pd.DataFrame({'id': test_id,
                               'type': result})

    submission.to_csv('submission.csv', index="False")
    # csv = open("result.csv","w")
    # tile_row="id, type\n"
    # csv.write(tile_row)


# Bagged  ExtraTrees for Classification
def extratree_class(X,Y):
    seed = 7
    num_trees = 100
    max_features = 3
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = ExtraTreesClassifier(n_estimators=num_trees, max_features=max_features)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("EXTRATREE ACC:   ",results.mean())


# Adaboosting for Classification
def Adaboosting(X,Y):
    seed = 7
    num_trees = 30
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("ADABOOSTING ACC: ",results.mean())


#stochastic gradient boosting
def stochastic_gradientboost(X,Y):
    seed = 7
    num_trees = 100
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
    results = model_selection.cross_val_score(model, X, Y, cv=kfold)
    print("STOCHASTIC_GRADIENT_BOOSTING:   ",results.mean())

def votingclass(estimators):
    ensemble = VotingClassifier(estimators)
    seed = 7
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)
    print("VOTING CLASS:    ",results.mean())
    print ("RESULTTTTT   ",results)


def main():
    bagginf_DecisionTree(X,Y)
    RandomForestClass(X,Y)
    extratree_class(X,Y)
    Adaboosting(X,Y)
    stochastic_gradientboost(X,Y)
    estimators=[]
    model1=DecisionTreeClassifier()
    model2=RandomForestClassifier()
    model3=ExtraTreesClassifier()
    model4 =AdaBoostClassifier()
    model5 = GradientBoostingClassifier()
    estimators.append(("decision",model1))
    estimators.append(("randomforest",model2))
    estimators.append(("extratree",model3))
    estimators.append(("adaboost",model4))
    estimators.append(("gradientboost",model5))
    votingclass(estimators)
    model3.fit(X,Y)
    print ("SCOREEEEEEEEEE   ",model3.score(X,Y))
    res = model3.predict(Xt)
    print ("RESSSSS  ",res)
    save_result(res)

if __name__=='__main__':
    main()

