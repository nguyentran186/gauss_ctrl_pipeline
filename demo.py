from ControlNetPipeline import GaussCtrlPipeline, GaussCtrlPipelineConfig

pipe = GaussCtrlPipeline(GaussCtrlPipelineConfig(), 'data/statue')
pipe.edit_images()

print("Ininitialize successfully")