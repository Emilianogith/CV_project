import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

"""
    model with hyerarchical fusion between Visual and Non_visual branches,
    based on the paper https://arxiv.org/pdf/2104.05485
"""

# Attention Module
class AttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionModule, self).__init__()
        self.Ws = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.Wc = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, h, hidden_states):

        Ws_h = self.Ws(h)
        scores = torch.matmul(hidden_states, self.Ws(h).unsqueeze(-1)).squeeze(-1)

        alpha = F.softmax(scores, dim=1)
        alpha_expanded = alpha.unsqueeze(1)
        h_c = torch.bmm(alpha_expanded, hidden_states).squeeze(1)

        concat_h_c_h = torch.cat([h_c, h], dim=1)
        v_attention = torch.tanh(self.Wc(concat_h_c_h))
        v_attention = self.dropout(v_attention)
        return v_attention



# Final Attention Module
class FinalAttentionModule(nn.Module):
    def __init__(self, hidden_dim):
        super(FinalAttentionModule, self).__init__()
        self.Ws = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)
        self.Wc = nn.Linear(hidden_dim * 4, hidden_dim, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, fused_features):
        batch_size, feature_dim = fused_features.shape


        fused_features_expanded = fused_features.unsqueeze(1)  # (batch_size, 1, feature_dim)
        Ws_fused = self.Ws(fused_features_expanded)  # (batch_size, 1, hidden_dim * 2)


        attention_scores = torch.bmm(fused_features_expanded, Ws_fused.transpose(1, 2))  # (batch_size, 1, 1)
        alpha = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, 1)


        context_vector = torch.bmm(alpha, fused_features_expanded)  # (batch_size, 1, hidden_dim * 2)
        context_vector = context_vector.squeeze(1)  # (batch_size, hidden_dim * 2)
        concat_attention = torch.cat([context_vector, fused_features], dim=-1)  # (batch_size, hidden_dim * 4)

        attention_output = self.Wc(concat_attention)  # (batch_size, hidden_dim)
        attention_output = self.dropout(attention_output)

        return attention_output


# Transformation Module
class TransformModule:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
            transforms.RandomRotation(10)
    
        ])

    def __call__(self, img):
        return self.transform(img)
    

# Visual Feature Extraction (VGG19-based CNN)
def build_cnn_module():
    vgg19 = models.vgg19(weights='DEFAULT')
    cnn = nn.Sequential(*list(vgg19.features.children())[:28]) # up to the 4th maxpooling layer

    # Freeze the parameters
    for param in cnn.parameters():
        param.requires_grad = False

    return cnn


# Visual Branch
class VisualBranch(nn.Module):
    def __init__(self, visual_input_shape, hidden_size=256):
        super(VisualBranch, self).__init__()
        self.cnn = build_cnn_module()
        self.gru = nn.GRU(512, hidden_size, batch_first=True)
        self.attention = AttentionModule(hidden_size)
        self.avg_pool = nn.AvgPool2d(kernel_size=(14, 14))
        self.transform_module = TransformModule()

    def forward(self, visual_input):

        batch_size, timesteps, C, H, W = visual_input.shape
        #print(visual_input.shape)

        # Reshape input to apply CNN on each frame: (batch_size * timesteps, C, H, W)
        visual_input = visual_input.reshape(batch_size * timesteps, C, H, W)
        visual_input = self.transform_module(visual_input) 
        cnn_features = self.cnn(visual_input)  # (batch_size * timesteps, features=512, 14, 14)

        pooled_features = self.avg_pool(cnn_features)
        pooled_features = pooled_features.reshape(batch_size, timesteps, -1)  # Reshape back: (batch_size, timesteps, features)

        x, h = self.gru(pooled_features)  # (batch_size, timesteps, hidden_size), h: (1,batch_size, hidden_size)

        #print('x: ',x.shape)
        #print('h: ',h.shape)

        x = self.attention(h.squeeze(0), x)  # Apply attention

        return x


# Non-Visual Branch
class NonVisualBranch(nn.Module):
    def __init__(self, hidden_size=256):
        super(NonVisualBranch, self).__init__()
        self.gru_pose = nn.GRU(64, hidden_size, batch_first=True)
        self.gru_trajectory = nn.GRU(hidden_size + 4, hidden_size, batch_first=True)
        self.gru_final = nn.GRU(hidden_size + 1, hidden_size, batch_first=True)
        self.attention = AttentionModule(hidden_size)

    def forward(self, pose_input, trajectory_input, speed_input):

        pose_features, _ = self.gru_pose(pose_input)

        concatenated = torch.cat([pose_features, trajectory_input], dim=-1)
        trajectory_features, _ = self.gru_trajectory(concatenated)
        final_concat = torch.cat([trajectory_features, speed_input], dim=-1)
        non_visual_features, h = self.gru_final(final_concat)

        #print('non_visual_features:',non_visual_features.shape)

        non_visual_features = self.attention(h.squeeze(0), non_visual_features)
        return non_visual_features

# Hybrid Fusion Network
class HybridFusionNetwork(nn.Module):
    def __init__(self, visual_input_shape, pose_input_shape, trajectory_input_shape, speed_input_shape, hidden_size=256):
        super(HybridFusionNetwork, self).__init__()
        self.visual_branch = VisualBranch(visual_input_shape, hidden_size)
        self.non_visual_branch = NonVisualBranch(hidden_size)
        self.final_attention = FinalAttentionModule(hidden_size)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, visual_input, pose_input, trajectory_input, speed_input):
        visual_features = self.visual_branch(visual_input)
        non_visual_features = self.non_visual_branch(pose_input, trajectory_input, speed_input)
        fused_features = torch.cat([visual_features, non_visual_features], dim=-1)

        fused_attention_output = self.final_attention(fused_features)

        output = torch.sigmoid(self.fc(fused_attention_output))
        return output



if __name__ == "__main__":
    visual_input_shape = (16, 3, 224, 224)  # (timesteps, channels, height, width)
    pose_input_shape = (16, 64)
    trajectory_input_shape = (16, 4)
    speed_input_shape = (16, 1)

    model = HybridFusionNetwork(visual_input_shape, pose_input_shape, trajectory_input_shape, speed_input_shape)

    #Display the total parameters of the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
