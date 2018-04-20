#%% LIBRARY IMPORTS

import numpy as np
import pandas as pd
import seaborn as sb
from pandas import ExcelWriter
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from sklearn.model_selection import train_test_split
import webbrowser

#%% DATA IMPORTS

dense_emea = pd.read_excel('dense_C.xlsx')
highvol_emea = pd.read_excel('high volume_C.xlsx')

#%% COST CATEGORIZATION OF COMMODITIES

# system board

comm225 = ['SYSTEM BOARDS']

# processor 

comm750 = ['PROCESSORS']

# hard drives, memory, power supplies

comm075 = ['HARDFILES SATA', 'HARDFILES SAS', 'HARDFILES SSD', 'MEMORY MODULES', 'POWER SUPPLIES','VOLTAGE REGULATORS / PADDLE CARD','BATTERIES','MEMORY OTHER', 'POWER BACKPLANES','RAID MEMORY','TAPE DRIVES', 'RAID BATTERY', 'MEMORY CARDS'] 

# all other parts

comm150 = ['NETWORK ADAPTERS', 'CONSUMABLE', 'SYSTEM CARDS', 'CABLES', 'MECHANICAL ASSEMBLIES', 'MIDPLANES', 'FANS & BLOWERS', 'RISERS', 'RAID ADAPTERS', 'GPU','DASD BACKPLANES', 'RAID SUPERCAP', 'OTHER BACKPLANES', 'SFP MODULES', 'CD-ROM / DVD-ROM', 'STORAGE ENCLOSURES', 'DASD / SCSI ADAPTERS', 'VIDEO ADAPTERS', 'NETWORK SWITCHES']


#%% CHOOSING THE DATABASE TO WORK ON

current = input('Type one of the following: 1. dense 2. highvol')

if current in ['dense'] :
    df1 = dense_emea
    
if current in ['highvol']:
    df1 = highvol_emea

#%% ADDING A PART COST COLUMN

partcostlist = list()
partcost = float()

for i in df1['Commodity']:
    
    partcost = 0
    
    if i in comm225:
        partcost = 2.25
    if i in comm750:
        partcost = 7.5
    if i in comm075:
        partcost = 0.75
    if i in comm150:
        partcost = 1.5
       
    partcostlist.append(partcost)
    
partcostlist = pd.DataFrame(partcostlist)
df1 = df1.assign(cost_of_part = partcostlist.values)  



#%% ADDING A SERVICE COST COLUMN

totalservicecostlist = list()
totalservicecost = float()
    
for j in range(0,df1.shape[0]):
    
    if df1['Service_Delivery_Type'].loc[j] in ['FOP']:
        totalservicecost = 0.5
        
    if df1['Service_Delivery_Type'].loc[j] in ['NPRA']:
        totalservicecost = 2.125
        
    if df1['Service_Delivery_Type'].loc[j] in ['CRU']:
        totalservicecost = 0.5 + df1['cost_of_part'][j] * df1['Part_Count'][j]
        
    if df1['Service_Delivery_Type'].loc[j] in ['ONS']:
        totalservicecost = 2 + df1['cost_of_part'][j] * df1['Part_Count'][j]
        
    if df1['Has the case been escalated to L3?'].loc[j] in ['Yes']:
        totalservicecost = totalservicecost + 10
    
    totalservicecostlist.append(totalservicecost)

totalservicecostlist = pd.DataFrame(totalservicecostlist)
df1 = df1.assign(total_service_cost = totalservicecostlist.values)

#%% DATA ANALYSIS    

# SIZE

print('\n Dimensions of ',current,'-EMEA: \n', df1.shape)

df1_rows = df1.shape[0]
df1_columns = df1.shape[1]

# COLUMNS

print('\n The ',current,'-EMEA dataset has the following attributes: \n')

for x in df1.columns:
    print(x, type(x))

# PRIMARY KEY TEST

print('\n Is the column a primary key? \n')

column_names = df1.columns

for y in column_names:
    if len(df1[y]) == len(set(df1[y])):
        print(y,' - Yes ')
    else:
        print(y,' - No ')

# COUNT OF UNIQUE INSTANCES

print('\n Number of unique instances in the column: \n')

for z in column_names:
    print(z,' - ',len(set(df1[z])))

# CATEGORICAL ATTRIBUTES AND CATEGORIES

print('\n For which columns are the values categorical? \n')

for k in column_names:
    if (len(set(df1[k]))>1 and len(set(df1[k]))<100):
        print(k)
        print('\n Count of ',k, '\n', Counter(df1[k]),'\n')
        level_counts = Counter(df1[k])
        plt.figure(figsize=(10,10))
        plt.title(k)
        df1[k].value_counts().plot(kind='barh')



#%% SEGREGATING CLAIMS BY SINGLE/MULTIPLE APPEARANCES

claim_count = Counter(df1['Claim_Nbr'])

single_claims = list()
multiple_claims = list()

for i in claim_count:
    if claim_count[i] > 1:
        multiple_claims.append(i)
        
for i in claim_count:
    if claim_count[i] == 1:
        single_claims.append(i)

#%% CREATING SEPARATE DATAFRAMES FOR SINGLE/MULTIPLE CLAIMS

single_services = []        
multiple_services = []

for i in range(0,len(df1['Claim_Nbr'])):
    if df1['Claim_Nbr'][i] in single_claims:
        single_services.append(df1.loc[i])
        
for i in range(0,len(df1['Claim_Nbr'])):
    if df1['Claim_Nbr'][i] in multiple_claims:
        multiple_services.append(df1.loc[i])
        
single_services = pd.DataFrame(single_services)
multiple_services = pd.DataFrame(multiple_services)

single_services = single_services.reset_index(drop=True)
multiple_services = multiple_services.reset_index(drop=True)

currentbadservicecost = np.sum(multiple_services['total_service_cost'])



#%% CREATING A NUMERIC LABEL FOR L3 IN THE SINGLE SERVICES DATABASE

level3label = int()
level3labellist = list()

for k in range(0,single_services.shape[0]):
    
    if single_services['Has the case been escalated to L3?'].loc[k] in ['No']:
        level3label = 0
        
    if single_services['Has the case been escalated to L3?'].loc[k] in ['Yes']:
        level3label = 1
        
    level3labellist.append(level3label)
        
level3labellist = pd.DataFrame(level3labellist)        
single_services = single_services.assign(L3_label = level3labellist.values)

#%% CREATING A NUMERIC LABEL FOR L3 IN THE MULTIPLE SERVICES DATABASE

level3label_2 = int()
level3labellist_2 = list()

for k in range(0,multiple_services.shape[0]):
    
    if multiple_services['Has the case been escalated to L3?'].loc[k] in ['No']:
        level3label_2 = 0
        
    if multiple_services['Has the case been escalated to L3?'].loc[k] in ['Yes']:
        level3label_2 = 1
        
    level3labellist_2.append(level3label_2)
        
level3labellist_2 = pd.DataFrame(level3labellist_2)        
multiple_services = multiple_services.assign(L3_label = level3labellist_2.values)

#%% REPLACING BLANK COMMODITIES WITH NONE

single_services.Commodity = single_services.Commodity.fillna('NONE')
multiple_services.Commodity = multiple_services.Commodity.fillna('NONE')


#%% IMPORTING MULTIPLE CLAIMS' FINAL SERVICE RECORDS

final_service_dense = pd.read_excel('multiple services - final service - dense.xlsx')
final_service_highvol = pd.read_excel('multiple services - final service - highvol.xlsx')

if current in ['dense'] :
    final_services = final_service_dense
    
if current in ['highvol']:
    final_services = final_service_highvol
    

#%% CREATING A NUMERIC LABEL FOR SERVICE TYPE IN SINGLE SERVICES DATABASE    

service_labels = int()
service_labels_list = list()

for k in range(0,single_services.shape[0]):
    
    if single_services['Service_Delivery_Type'].loc[k] in ['FOP']:
        service_labels = 1
    if single_services['Service_Delivery_Type'].loc[k] in ['CRU']:
        service_labels = 2    
    if single_services['Service_Delivery_Type'].loc[k] in ['ONS']:
        service_labels = 3
    if single_services['Service_Delivery_Type'].loc[k] in ['NPRA']:
        service_labels = 4
        
    service_labels_list.append(service_labels)
        
service_labels_list = pd.DataFrame(service_labels_list)        
single_services = single_services.assign(Service_Labels = service_labels_list.values)

#%% CREATING A NUMERIC LABEL FOR SERVICE TYPE IN MULTIPLE SERVICES DATABASE    

service_labels_2 = int()
service_labels_list_2 = list()

for k in range(0,multiple_services.shape[0]):
    
    if multiple_services['Service_Delivery_Type'].loc[k] in ['FOP']:
        service_labels_2 = 1
    if multiple_services['Service_Delivery_Type'].loc[k] in ['CRU']:
        service_labels_2 = 2    
    if multiple_services['Service_Delivery_Type'].loc[k] in ['ONS']:
        service_labels_2 = 3
    if multiple_services['Service_Delivery_Type'].loc[k] in ['NPRA']:
        service_labels_2 = 4
        
    service_labels_list_2.append(service_labels_2)
        
service_labels_list_2 = pd.DataFrame(service_labels_list_2)        
multiple_services = multiple_services.assign(Service_Labels = service_labels_list_2.values)

#%% CREATING COLUMNS IN MULTIPLE SERVICES FOR PREDICTED LABEL AND PROBABILITIES

multiple_services['Predicted_Service'] = pd.Series()
multiple_services['Prob_FOP'] = pd.Series()
multiple_services['Prob_CRU'] = pd.Series()
multiple_services['Prob_ONS'] = pd.Series()
multiple_services['Prob_NPRA'] = pd.Series()

#%% CREATING A NUMERIC LABEL FOR SERVICE TYPE IN FINAL SERVICES DATABASE    

service_labels_3 = int()
service_labels_list_3 = list()

for k in range(0,final_services.shape[0]):
    
    if final_services['Service_Delivery_Type'].loc[k] in ['FOP']:
        service_labels_3 = 1
    if final_services['Service_Delivery_Type'].loc[k] in ['CRU']:
        service_labels_3 = 2    
    if final_services['Service_Delivery_Type'].loc[k] in ['ONS']:
        service_labels_3 = 3
    if final_services['Service_Delivery_Type'].loc[k] in ['NPRA']:
        service_labels_3 = 4
        
    service_labels_list_3.append(service_labels_3)
        
service_labels_list_3 = pd.DataFrame(service_labels_list_3)        
final_services = final_services.assign(Service_Labels = service_labels_list_3.values)

#%% CREATING COLUMNS IN MULTIPLE SERVICES FOR PREDICTED LABEL AND PROBABILITIES

multiple_services['Predicted_Service'] = pd.Series()
multiple_services['Prob_FOP'] = pd.Series()
multiple_services['Prob_CRU'] = pd.Series()
multiple_services['Prob_ONS'] = pd.Series()
multiple_services['Prob_NPRA'] = pd.Series()



#%% PERFECT SERVICE DATAFRAME (1 SERVICE, UNESCALATED)

perfect_services = []

for m in range(0,len(single_services['L3_label'])):
    if single_services['L3_label'].loc[m] == 0:
        perfect_services.append(single_services.loc[m])
        
perfect_services = pd.DataFrame(perfect_services)
perfect_services = perfect_services.reset_index(drop=True)

#%% TRAINING MODEL ON PERFECT SERVICES AND PREDICTING ON MULTIPLE SERVICES AND FINAL SERVICES

ps = perfect_services
ms = multiple_services
fs = final_services

age_count = len(set(ps['Month_in_Service']))
machine_count = len(set(ps['Machine_Type']))
country_count = len(set(ps['Country_Code']))
commodity_count = len(set(ps['Commodity']))

ps_rows = ps.shape[0]
ps_columns = ps.shape[1]

default_parameters = input('Build model with default parameters? Type yes or no')

if default_parameters in ['yes']:
    test_data_fraction = 0.25
    train_batch_size = 10
    train_num_epochs = 1000
    train_num_steps = 1000
    hidden_layer_units = [2,2]
    
if default_parameters in ['no']:
    test_data_fraction = float(input('Fraction of test data:'))
    train_batch_size = int(input('Batch size for training:'))
    train_num_epochs = int(input('Number of training epochs:'))
    
    num_of_layers = int(input('Enter number of layers:'))
    
    hidden_layer_units = list()
    
    for i in range (1,num_of_layers+1):    
        print('For layer ',i,':')
        current_layer_neurons = int(input('No of neurons in layer:'))
        hidden_layer_units.append(current_layer_neurons)
        
    train_num_steps = int(input('Number of training steps:'))

cols_to_norm = ['Month_in_Service']
ps[cols_to_norm] = ps[cols_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

age = tf.feature_column.numeric_column('Month_in_Service')

country = tf.feature_column.categorical_column_with_hash_bucket('Country_Code',hash_bucket_size=country_count)
commodity = tf.feature_column.categorical_column_with_hash_bucket('Commodity',hash_bucket_size=commodity_count)
machine = tf.feature_column.categorical_column_with_hash_bucket('Machine_Type',hash_bucket_size=machine_count)

embedded_country_col = tf.feature_column.embedding_column(country,dimension=country_count)
embedded_commodity_col = tf.feature_column.embedding_column(commodity,dimension=commodity_count)
embedded_machine_col = tf.feature_column.embedding_column(machine,dimension=machine_count)

feat_cols = []
features_data_list = []

default_inputs = input('Use the default input columns? Enter yes or no')

if default_inputs in ['yes']:
    
    feat_cols = [age,embedded_country_col,embedded_commodity_col,embedded_machine_col]
    features_data = ps[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    features_data_2 = ms[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    features_data_3 = fs[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    
if default_inputs in ['no']:
    
    feat_cols = [age,embedded_country_col,embedded_commodity_col,embedded_machine_col]
    features_data = ps[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    features_data_2 = ms[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    features_data_3 = fs[['Month_in_Service','Country_Code','Commodity','Machine_Type']]
    
    input_dict = {0:'Month_in_Service',1:'Country_Code',2:'Commodity',3:'Machine_Type'}
    input_data_dict = {0:age,1:embedded_country_col,2:embedded_commodity_col,3:embedded_machine_col}
    
    print('Following inputs are available: \n')
    
    for i in input_dict:
        print(i,input_dict[i])
    
    multiple_unwanted_inputs = []
    
    for remove_iter in range(0,100):
        
        remove_input = input('Do you want to remove an input?')
        
        if remove_input in ['yes']:
            
            
            unwanted_input = int(input('Enter the number of the input you want to remove:'))
            del features_data[input_dict[unwanted_input]]
            del features_data_2[input_dict[unwanted_input]]
            del features_data_3[input_dict[unwanted_input]]
            
            multiple_unwanted_inputs.append(unwanted_input)
            
        if remove_input in ['no']:
            break
    
    multiple_unwanted_inputs.sort()
    
    for each_unwanted in multiple_unwanted_inputs:
        
        feat_cols[each_unwanted]='need to delete this'
        
    for each in range(0,feat_cols.count('need to delete this')):
        feat_cols.remove('need to delete this')


labels_data = ps['Service_Labels']

accuracies_list = []

#for d in range(0,10):
        
x_train, x_test, y_train, y_test = train_test_split(features_data,labels_data,test_size=test_data_fraction)
input_func = tf.estimator.inputs.pandas_input_fn(x=x_train,y=y_train,batch_size=train_batch_size,num_epochs=train_num_epochs,shuffle=True)
dnn_model = tf.estimator.DNNClassifier(hidden_units=hidden_layer_units,feature_columns=feat_cols,n_classes=5)

dnn_model.train(input_fn = input_func, steps=train_num_steps)
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=x_test,y=y_test,batch_size=10, num_epochs=1, shuffle=False)
results = dnn_model.evaluate(eval_input_func)
print(results['accuracy'])
#    accuracies_list.append(results['accuracy'])
    
#avg_accuracy = np.mean(accuracies_list)

#plt.figure(figsize = (8,8))
#plt.plot(range(0,10),accuracies_list)
#plt.title('Accuracy variation over iterations')
#plt.xlabel('Iteration')
#plt.ylabel('Accuracy')
#plt.ylim((0,1))
    
x_pred = features_data_2
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=x_pred,batch_size=10,num_epochs=1,shuffle=False)

predictions = dnn_model.predict(pred_input_func)

predictions_list = []
prob_fop_list = []
prob_cru_list = []
prob_ons_list = []
prob_npra_list = []

for x in list(predictions):
    predictions_list.append(x['class_ids'][0])
    prob_fop_list.append(x['probabilities'][1])
    prob_cru_list.append(x['probabilities'][2])
    prob_ons_list.append(x['probabilities'][3])
    prob_npra_list.append(x['probabilities'][4])
    
#    print(x['class_ids'][0])
    
predictions_list = pd.DataFrame(predictions_list)
prob_fop_list = pd.DataFrame(prob_fop_list)
prob_cru_list = pd.DataFrame(prob_cru_list)
prob_ons_list = pd.DataFrame(prob_ons_list)
prob_npra_list = pd.DataFrame(prob_npra_list)

multiple_services = multiple_services.assign(Predicted_Service = predictions_list.values)
multiple_services = multiple_services.assign(Prob_FOP = prob_fop_list.values)
multiple_services = multiple_services.assign(Prob_CRU = prob_cru_list.values)
multiple_services = multiple_services.assign(Prob_ONS = prob_ons_list.values)
multiple_services = multiple_services.assign(Prob_NPRA = prob_npra_list.values)


correct_predictions = 0

for y in range(0,len(multiple_services)):
        if multiple_services['Service_Labels'].loc[y] == multiple_services['Predicted_Service'].loc[y]:
            correct_predictions = correct_predictions + 1
        if multiple_services['Service_Labels'].loc[y] != multiple_services['Predicted_Service'].loc[y]:
            correct_predictions = correct_predictions + 0
            
validation_accuracy = correct_predictions / len(multiple_services)



x_pred_3 = features_data_3
pred_input_func_3 = tf.estimator.inputs.pandas_input_fn(x=x_pred_3,batch_size=10,num_epochs=1,shuffle=False)

predictions_3 = dnn_model.predict(pred_input_func_3)

predictions_list_3 = []
prob_fop_list_3 = []
prob_cru_list_3 = []
prob_ons_list_3 = []
prob_npra_list_3 = []

for q in list(predictions_3):
    predictions_list_3.append(q['class_ids'][0])
    prob_fop_list_3.append(q['probabilities'][1])
    prob_cru_list_3.append(q['probabilities'][2])
    prob_ons_list_3.append(q['probabilities'][3])
    prob_npra_list_3.append(q['probabilities'][4])
    
#    print(x['class_ids'][0])
    
predictions_list_3 = pd.DataFrame(predictions_list_3)
prob_fop_list_3 = pd.DataFrame(prob_fop_list_3)
prob_cru_list_3 = pd.DataFrame(prob_cru_list_3)
prob_ons_list_3 = pd.DataFrame(prob_ons_list_3)
prob_npra_list_3 = pd.DataFrame(prob_npra_list_3)

final_services = final_services.assign(Predicted_Service = predictions_list_3.values)
final_services = final_services.assign(Prob_FOP = prob_fop_list_3.values)
final_services = final_services.assign(Prob_CRU = prob_cru_list_3.values)
final_services = final_services.assign(Prob_ONS = prob_ons_list_3.values)
final_services = final_services.assign(Prob_NPRA = prob_npra_list_3.values)


correct_predictions_3 = 0

for r in range(0,len(final_services)):
        if final_services['Service_Labels'].loc[r] == final_services['Predicted_Service'].loc[r]:
            correct_predictions_3 = correct_predictions_3 + 1
        if final_services['Service_Labels'].loc[r] != final_services['Predicted_Service'].loc[r]:
            correct_predictions_3 = correct_predictions_3 + 0
            
validation_accuracy_3 = correct_predictions_3 / len(final_services)

#%% FINDING COSTS AFTER PREDICTION

predictedservicecostlist = list()
predictedservicecost = float()
    
for c in range(0,final_services.shape[0]):
    
    if final_services['Predicted_Service'].loc[c] == 1 :
        predictedservicecost = 0.5
        
    if final_services['Predicted_Service'].loc[c] == 4 :
        predictedservicecost = 2.125
        
    if final_services['Predicted_Service'].loc[c] == 2 :
        predictedservicecost = 0.5 + final_services['cost_of_part'][c] * final_services['Part_Count'][c]
        
    if final_services['Predicted_Service'].loc[c] == 3 :
        predictedservicecost = 2 + final_services['cost_of_part'][c] * final_services['Part_Count'][c]
        
    if final_services['Has the case been escalated to L3?'].loc[c] in ['Yes'] :
        predictedservicecost = predictedservicecost + 10
    
    predictedservicecostlist.append(predictedservicecost)

predictedservicecostlist = pd.DataFrame(predictedservicecostlist)
final_services = final_services.assign(predicted_service_cost = predictedservicecostlist.values)

predictedbadservicecost = np.sum(final_services['predicted_service_cost'])
#%% PREDICTIONS WHEN A NEW CALL ARRIVES AT CUSTOMER CARE

new_service = pd.read_excel('new service.xlsx')

#print('Enter the details of the current call:')
#
#c0 = input('Enter country:')
#c1 = input('Enter machine type:')
#c2 = input('Enter commodity if known, or enter NONE:')
#c3 = int(input('Enter month in service:'))
#
#new_service = pd.DataFrame(index = [0,1,2,3,4,5,6,7,8,9], columns = multiple_services.columns)
#    
#new_service.set_value(0,'Country_Code', c0)
#new_service.set_value(0,'Machine_Type', c1)
#new_service.set_value(0,'Commodity', c2)
#new_service.set_value(0,'Month_in_Service', c3)

features_data_new = new_service[['Country_Code','Machine_Type','Month_in_Service','Commodity']]
x_pred_new = features_data_new

pred_input_func_new = tf.estimator.inputs.pandas_input_fn(x=x_pred_new,num_epochs=1,shuffle=False)

predictions_new = dnn_model.predict(pred_input_func_new)

predictions_list_new = []
prob_fop_list_new = []
prob_cru_list_new = []
prob_ons_list_new = []
prob_npra_list_new = []

for z in list(predictions_new):
    predictions_list_new.append(z['class_ids'][0])
    prob_fop_list_new.append(z['probabilities'][1])
    prob_cru_list_new.append(z['probabilities'][2])
    prob_ons_list_new.append(z['probabilities'][3])
    prob_npra_list_new.append(z['probabilities'][4])
        
predictions_list_new = pd.DataFrame(predictions_list_new)
prob_fop_list_new = pd.DataFrame(prob_fop_list_new)
prob_cru_list_new = pd.DataFrame(prob_cru_list_new)
prob_ons_list_new = pd.DataFrame(prob_ons_list_new)
prob_npra_list_new = pd.DataFrame(prob_npra_list_new)

new_service = new_service.assign(Predicted_Service = predictions_list_new.values)
new_service = new_service.assign(Prob_FOP = prob_fop_list_new.values)
new_service = new_service.assign(Prob_CRU = prob_cru_list_new.values)
new_service = new_service.assign(Prob_ONS = prob_ons_list_new.values)
new_service = new_service.assign(Prob_NPRA = prob_npra_list_new.values)

#%% FINDING CURRENT TOTAL COSTS BY MACHINE BY COUNTRY

# Not making it more specific because then there are too few instances in each group

groupedstuff = df1.groupby(['Country_Code','Machine_Type','Month_in_Service','Commodity'])
groupedstuff2 = df1.groupby(['Country_Code','Machine_Type','Month_in_Service','Commodity'])['total_service_cost'].sum()

grouped_inputs = pd.DataFrame(index = np.arange(0,len(groupedstuff2)), columns = ['Country_Code','Machine_Type','Months_in_Service','Commodity'])
countryy = []
machinee = []
commodityy = []
agee = []

for name, group in groupedstuff:
    countryy.append(name[0])
    machinee.append(name[1])
    agee.append(name[2])
    commodityy.append(name[3])

countryy = pd.DataFrame(countryy)
machinee = pd.DataFrame(machinee)
agee = pd.DataFrame(agee)
commodityy = pd.DataFrame(commodityy)

grouped_inputs = grouped_inputs.assign(Country_Code = countryy.values)
grouped_inputs = grouped_inputs.assign(Machine_Type = machinee.values)
grouped_inputs = grouped_inputs.assign(Month_in_Service = agee.values)
grouped_inputs = grouped_inputs.assign(Commodity = commodityy.values)
#grouped_currentcost = grouped_currentcost.assign(Cost = groupedstuff2.values)

writer = ExcelWriter('grouped inputs.xlsx')
grouped_inputs.to_excel(writer, 'Sheet1')
writer.save()

#%% LIST OF PREDICTIONS FOR GROUPED INPUTS

features_data_categories = grouped_inputs[['Country_Code','Machine_Type','Month_in_Service','Commodity']]
x_pred_categories = features_data_categories

pred_input_func_categories = tf.estimator.inputs.pandas_input_fn(x=x_pred_categories,num_epochs=1,shuffle=False)

predictions_categories = dnn_model.predict(pred_input_func_categories)

predictions_list_categories = []
prob_fop_list_categories = []
prob_cru_list_categories = []
prob_ons_list_categories = []
prob_npra_list_categories = []

for p in list(predictions_categories):
    predictions_list_categories.append(p['class_ids'][0])
    prob_fop_list_categories.append(p['probabilities'][1])
    prob_cru_list_categories.append(p['probabilities'][2])
    prob_ons_list_categories.append(p['probabilities'][3])
    prob_npra_list_categories.append(p['probabilities'][4])
        
predictions_list_categories = pd.DataFrame(predictions_list_categories)
prob_fop_list_categories = pd.DataFrame(prob_fop_list_categories)
prob_cru_list_categories = pd.DataFrame(prob_cru_list_categories)
prob_ons_list_categories = pd.DataFrame(prob_ons_list_categories)
prob_npra_list_categories = pd.DataFrame(prob_npra_list_categories)

grouped_inputs = grouped_inputs.assign(Predicted_Service = predictions_list_categories.values)
grouped_inputs = grouped_inputs.assign(Prob_FOP = prob_fop_list_categories.values)
grouped_inputs = grouped_inputs.assign(Prob_CRU = prob_cru_list_categories.values)
grouped_inputs = grouped_inputs.assign(Prob_ONS = prob_ons_list_categories.values)
grouped_inputs = grouped_inputs.assign(Prob_NPRA = prob_npra_list_categories.values)

writer = ExcelWriter('grouped inputs.xlsx')
grouped_inputs.to_excel(writer, 'Sheet1')
writer.save()


#%%

print("Accuracy when buildng the model: ",validation_accuracy)
print("Acuuracy when applied on multiple claim data: ", validation_accuracy_3)

print("Current Cost in Bad Service: ",currentbadservicecost)
print("Cost of Predicted Service: ", predictedbadservicecost)

#%% TEMPLATE TO EXPORT A DF TO AN EXCEL FILE

writer = ExcelWriter('new service.xlsx')
new_service.to_excel(writer, 'Sheet1')
writer.save()

writer = ExcelWriter('single services.xlsx')
single_services.to_excel(writer, 'Sheet1')
writer.save()

writer = ExcelWriter('multiple services.xlsx')
multiple_services.to_excel(writer, 'Sheet1')
writer.save()

writer = ExcelWriter('perfect services.xlsx')
perfect_services.to_excel(writer, 'Sheet1')
writer.save()

writer = ExcelWriter('final services.xlsx')
final_services.to_excel(writer, 'Sheet1')
writer.save()

#webbrowser.open('new service.xlsx')
#webbrowser.open('final services.xlsx')