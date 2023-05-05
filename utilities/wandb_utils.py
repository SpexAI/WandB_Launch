import wandb

'''
Init Wandb object for tracking
Input:
+ project_name: name for the project you want to do
+ anonymus_status: status anonymus for wandb
Output:
+run: Wandb runtime object
'''

def wandb_init(project_name = "flowerDetection", 
               anonymus_status = 'must', config = None):

    try:

        #wandb.login(anonymous=anonymus_status)
        run = wandb.init(project = project_name, config = config, entity="spexai", job_type="train")

        return run

    except Exception as error:
        
        print('Raise this error: ' + repr(error))

        return None

'''
Wandb artifact for saving training file on Wandb
Input:
+ run: Wandb runtime object
+ model_output_dir: path to save model
+ dataset_name: name for dataset
'''

def wandb_artifact(run, 
                   model_output_dir = 'model.pth',
                   data_name = 'model_data'):

    try:

        model_data = wandb.Artifact(data_name, type="model")
        model_data.add_file(model_output_dir)
        run.log_artifact(model_data)
        run.finish()

    except Exception as error:
        
        print('Raise this error: ' + repr(error))