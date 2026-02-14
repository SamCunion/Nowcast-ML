#queries the given model in a standardised way, and returns a standardised output
import torch
import numpy as np

#takes loaded model, the input to the model (probably the stack [1, 4, 120, 240]), and whether to perform GRAD-CAM
#returns probability of tornado, intensity probs sum([7]) = 1, and [Heads, 120, 240] GRAD-CAM if specified else None
#input stack if querying torcast 2, or DBZ, VEL and RHOHV if querying torcast 0 or 1. with_grad determines if gradcam is also output
def Query_Model(standard_torcast_model, stack=None, DBZ=None, VEL=None, RHOHV=None, with_grad=False):

    #get apply hooks
    if (with_grad):
        target_heads = standard_torcast_model.gradcam_targets #each model's architecture specifies their gradcam targets
        target_layers = []
        activations = []
        gradients = []
        for head in target_heads:
            target_layers.append(next(layer for layer in reversed(getattr(standard_torcast_model, head)) if isinstance(layer, torch.nn.Conv2d))) #gets final conv layer in head


        for i in range(len(target_layers)):
            #register the hooks
            layer = target_layers[i]
            def forward_hook(module, input, output):
                activations.append(output.detach())
            
            def backward_hook(module, input, output):
                gradients.append(output[0].detach())

            layer.register_forward_hook(forward_hook)
            layer.register_full_backward_hook(backward_hook)

    standard_torcast_model.eval()
    
    if (stack != None): #stacked head input
        prob_logit = standard_torcast_model(stack)
    else: #separate head input
        prob_logit = standard_torcast_model(DBZ, VEL, RHOHV)

    #get outputs from both heads
    tornado_prob = torch.nn.functional.sigmoid(prob_logit.squeeze()).detach().numpy()

    
    if (with_grad):
        
        #backward passes from the probability head
        standard_torcast_model.zero_grad()
        prob_logit.backward()
        cams = []

        for i in range(len(gradients)):
            grad = gradients[i]
            acts = activations[i]
            #performs standard grad-cam process
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = torch.relu(cam)
            #resize for output
            cam = torch.nn.functional.interpolate(cam, size=(120, 240), mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()
            cam = (cam - cam.min()) / (cam.max() + 1e-9) #normalise

            cams.append(cam)

        return tornado_prob, cams
    
    return tornado_prob, None