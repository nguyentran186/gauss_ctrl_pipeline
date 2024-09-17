from ControlNetPipeline import GaussCtrlPipeline, GaussCtrlPipelineConfig

pipe = GaussCtrlPipeline(GaussCtrlPipelineConfig(), '/home/ubuntu/workspace/bhrc/nam/gaussctrl/data/data/statue')
pipe.render_reverse()
pipe.edit_images()

print("Ininitialize successfully")