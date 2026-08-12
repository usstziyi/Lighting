from torchview import draw_graph
from torch import nn
import torch 
from torchvision.models import resnet18
from model import ResNet


def export_mlp_graph():
    class MLP(nn.Module):
        def __init__(self, in_features=128, hidden_features=64, out_features=10):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    model = MLP()
    # device='meta' -> no memory is consumed for visualization
    model_graph = draw_graph(
        model, 
        input_size=(1, 128), 
        device='meta',
        expand_nested=True, # 展开嵌套模块
        hide_module_functions=True, # 隐藏模块函数
        save_graph=False,
    )
    # 提高渲染 DPI（默认 96），图片更清晰；也可改成 svg 得到无限清晰矢量图
    model_graph.visual_graph.graph_attr['dpi'] = '300'
    model_graph.visual_graph.render(
        filename='mlp_graph',
        directory='./viz',
        format='png', 
        cleanup=True
    )


def export_resnet18_graph():
    model = resnet18(num_classes=10)
    model_graph = draw_graph(
        model, 
        input_size=(1, 3, 32, 32), 
        device='meta', 
        expand_nested=True, # 展开嵌套模块
        hide_module_functions=True, # 隐藏模块函数
        save_graph=False,
    )
    # 提高渲染 DPI（默认 96），图片更清晰；也可改成 svg 得到无限清晰矢量图
    model_graph.visual_graph.graph_attr['dpi'] = '300'
    model_graph.visual_graph.render(
        filename='resnet18_graph',
        directory='./viz',
        format='png', 
        cleanup=True
    )


def export_manual_resnet18_graph():
    model = ResNet(num_classes=10)
    model_graph = draw_graph(
        model, 
        input_size=(1, 3, 32, 32), 
        device='meta', 
        expand_nested=True, # 展开嵌套模块
        hide_module_functions=True, # 隐藏模块函数
        save_graph=False,
    )
    # 提高渲染 DPI（默认 96），图片更清晰；也可改成 svg 得到无限清晰矢量图
    model_graph.visual_graph.graph_attr['dpi'] = '300'
    model_graph.visual_graph.render(
        filename='manual_resnet18_graph',
        directory='./viz',
        format='png', 
        cleanup=True
    )


def main():
    export_mlp_graph()
    export_resnet18_graph()
    export_manual_resnet18_graph()

if __name__ == '__main__':
    main()



