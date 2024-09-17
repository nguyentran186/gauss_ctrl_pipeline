from ControlNetPipeline import GaussCtrlPipeline, GaussCtrlPipelineConfig

pipe = GaussCtrlPipeline(GaussCtrlPipelineConfig(), 'data/statue')
pipe.render_reverse()

print("Ininitialize successfully")