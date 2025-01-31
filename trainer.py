import torch
from tqdm import tqdm
from model.metric import *
from math import *

class Trainer:
    def __init__(self,model,criterion,opt,args,config,data_loader,eval_data_loader,lr_scheduler):
        # choose GPU
        self.device = args.gpu
        print(f'On GPU {self.device}')

        # load modules
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.opt = opt
        self.data_loader = data_loader
        self.eval_data_loader = eval_data_loader
        self.val_targets = torch.tensor(eval_data_loader.dataset.targets,dtype=torch.int64)
        self.lr_scheduler = lr_scheduler

        # load hyper-parameters
        self.args = args
        self.config = config
        self.num_class = self.val_targets.max().item()+1

    def run(self):
        pbar = tqdm(range(1, self.config["trainer"]["epochs"] + 1))
        for epoch in pbar:
            # train model
            total_loss = self._train_epoch(epoch)
            pbar.set_postfix({"training_loss": sum(total_loss)/len(total_loss)})

            # test model
            if epoch%40==0:
                print(f'================ Epoch: {epoch:03d} ================')
                self._test_epoch(epoch)

            # update learning rate
            self.lr_scheduler.step()

    def _train_epoch(self,epoch):
        self.model.train()
        self.model._hook_before_iter()
        self.criterion._hook_before_epoch(epoch)

        total_loss = []
        for _,(data,target) in enumerate(self.data_loader):
            data,target = data.to(self.device),target.to(self.device)
            self.opt.zero_grad()

            output = self.model(data)
            extra_info = {
                "num_particle"  : len(self.model.backbone.logits)   ,
                "logits"        : self.model.backbone.logits        ,
                "weights"       : self.model.backbone.linears
            }
            loss = self.criterion(
                epoch       = epoch     ,
                x           = output    ,
                y           = target    ,
                extra_info  = extra_info
            )
            loss.backward()

            self.opt.step()
            total_loss.append(loss.item())

        return total_loss

    def _test_epoch(self,epoch):
        self.model.eval()
        output = torch.empty(0, self.num_class, dtype=torch.float32)
        uncertainty = torch.empty(0, dtype=torch.float32)
        uncertainty_mcp = torch.empty(0, dtype=torch.float32)
        uncertainty_evi = torch.empty(0, dtype=torch.float32)

        for _,(data,_) in enumerate(self.eval_data_loader):
            data = data.to(self.device)

            with torch.no_grad():
                o = self.model(data)
                u = -torch.sum(F.softmax(o, dim=-1) * F.log_softmax(o, dim=-1), dim=-1)
                u_mcp = 1 - torch.max(F.softmax(o, dim=-1), dim=-1).values
                u_evi = o.shape[-1] / (torch.sum(F.relu(o), dim=-1) + o.shape[-1])

            output = torch.cat([output, o.detach().cpu()], dim=0)
            uncertainty = torch.cat([uncertainty, u.detach().cpu()])
            uncertainty_mcp = torch.cat([uncertainty_mcp, u_mcp.detach().cpu()])
            uncertainty_evi = torch.cat([uncertainty_evi, u_evi.detach().cpu()])

        pred = torch.argmax(output,dim=1)
        correct = (pred==self.val_targets)
        ACC(correct, self.val_targets, region=self.config['trainer']['region'])

        print('\n Entropy:')
        AUCECE(correct, uncertainty, self.val_targets, region=self.config['trainer']['region'])

        print("\n MCP:")
        AUCECE(correct, uncertainty_mcp, self.val_targets, region=self.config['trainer']['region'])

        print("\n Evidential:")
        AUCECE(correct, uncertainty_evi, self.val_targets, region=self.config['trainer']['region'])

        FHR(pred,self.val_targets,self.num_class)

        # CIAFR10_bird2plane(pred,self.val_targets)
        # CIAFR10_car2animal(pred,self.val_targets)
