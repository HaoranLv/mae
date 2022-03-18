import os
import time
import torch
import numpy as np
import models_vit
from PIL import Image
imgsz = 640
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

def prepare_img(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.
    assert img.shape == (224, 224, 3)
    
    img = img - imagenet_mean
    img = img / imagenet_std
    return img

def model_fn(model_dir):
    arch='vit_base_patch16'
    # load model
    model_dir=os.path.join(model_dir, 'checkpoint.pth')
    checkpoint = torch.load(model_dir, map_location='cuda:0')
    num_classes=checkpoint['model']['head.weight'].shape[0]
    # build model
    model = models_vit.__dict__[arch](
        num_classes=num_classes,
        drop_path_rate=0,
        global_pool=False,
    )
    interpolate_pos_embed(model, checkpoint['model'])
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    return model

def input_fn(request_body, request_content_type):
#     print('[DEBUG] request_body:', type(request_body))
#     print('[DEBUG] request_content_type:', request_content_type)
    
    """An input_fn that loads a pickled tensor"""
    if request_content_type == 'application/python-pickle':
        from six import BytesIO
        return torch.load(BytesIO(request_body))
    elif request_content_type == 'application/x-npy':
        from io import BytesIO
        np_bytes = BytesIO(request_body)
        return np.load(np_bytes, allow_pickle=True)
    else:
        # Handle other content-types here or raise an Exception
        # if the content type is not supported.  
        return request_body


def predict_fn(input_data, model):
    img = Image.fromarray(cv2.cvtColor(input_data,cv2.COLOR_BGR2RGB))  
    img = prepare_img(img)
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)
    x = model.forward_features(x.float())
    return x.squeeze(dim=0)


# def output_fn(prediction, content_type):
#     pass


if __name__ == '__main__':
    import cv2
    input_data = cv2.imread('../../000000.jpg')
    model = model_fn('../')
    result = predict_fn(input_data, model)