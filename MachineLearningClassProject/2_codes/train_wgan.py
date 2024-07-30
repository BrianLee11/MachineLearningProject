import torch
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter
from c_wgan import Generator, Critic, initialize_weight
from utill import gradient_penalty, Dataset, load_fashion_mnist, embedder
import matplotlib.pyplot as plt
import os
import shutil

if os.path.exists('logs'):
    shutil.rmtree('logs')
# Hyperparameters etc
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

load = True
lr = 8e-5 # learning rate

img_size = [28, 28]
img_channel = 1
z_dim = 10
n_epoch = 20
critic_features = 32
generator_features = 32+8
critic_iteration = 5
lambda_gp= 5


data_dir = 'Herald_square_data'
saved_loc = f'save_model/cwgan_z_{z_dim}.pt'
n_cond = 10
batch_size = 64

train_loader,  test_loader = load_fashion_mnist(batch_size)
# initialize gen and disc/critic
gen = Generator(z_dim, img_channel, generator_features, n_cond = n_cond).to(device)
critic = Critic(img_channel, critic_features, 
                img_size =img_size , 
                n_cond = n_cond).to(device)

print("Num params of generator: ", sum(p.numel() for p in gen.parameters()))
print("Num params of critic: ", sum(p.numel() for p in critic.parameters()))


opt_gen = optim.Adam(gen.parameters(), lr=lr, betas = (0.5,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=lr, betas = (0.5,0.9))
# schedulers are optional


initialize_weight(gen)
initialize_weight(critic)
writer_real = SummaryWriter(f"logs/real")
writer_fake = SummaryWriter(f"logs/fake")
step = 0

gen.train()
critic.train()

best_loss_critic = -1e5
best_loss_gen = 1e5

loss_gen_list, loss_critic_list = [], []          
for epoch in range(n_epoch):
    
    if epoch!=0 and epoch % 1 ==0:
        print("=============== model is saved ====================")
        torch.save({
            'gen':gen.state_dict(),
            'critic':critic.state_dict(),
             'opt_gen':opt_gen.state_dict(),
             'opt_critic':opt_critic.state_dict(),
             'epoch':epoch,
             'loss_critc':loss_critic_list,
             'loss_gen':loss_gen_list,
            }, saved_loc)
            
        
    loss_gen_sum, loss_critic_sum = 0,0
    
    for batch_idx, (real, cond) in enumerate(train_loader):
     
        cond = embedder(cond) 
        cond = cond.to(device, dtype = torch.float)
        real = real.to(device, dtype = torch.float)
        batch_size_now = real.shape[0]
       
        # Train Critic: max E[critic(real)] - E[critic(fake)]
        for _ in range(critic_iteration):
            noise = torch.randn(batch_size_now, z_dim, 1, 1).to(device)
            fake = gen(noise, cond)
            
            critic_real = critic(real, cond).reshape(-1)
            critic_fake = critic(fake, cond).reshape(-1)
            gp = gradient_penalty(critic, real, fake, cond, device = device)
            loss_critic =( -(torch.mean(critic_real) - torch.mean(critic_fake))
            + lambda_gp*gp)
            opt_critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()
           
        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        gen_fake = critic(fake, cond).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()
        
        loss_gen_sum += loss_gen.item()
        loss_critic_sum += loss_critic.item()
        # Print losses occasionally and print to tensorboard
        
        if batch_idx % 100 == 0 and batch_idx > 0:
            gen.eval()
            #critic.eval()
            print(
                f"Epoch [{epoch}/{n_epoch}] Batch {batch_idx}/{len(train_loader)} \
                  Loss D: {loss_critic:.4f}, loss G: {loss_gen:.4f}")
    
            with torch.no_grad():
                fake = gen(noise, cond)
                
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(
                    real[:32], normalize=True
                )
                img_grid_fake = torchvision.utils.make_grid(
                    fake[:32], normalize=True
                )
                
                
                writer_real.add_image("Real", img_grid_real, global_step=step)
                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                
                
                plt.figure(figsize = (4,18))
                fakes = fake[:16]
                for j in range(16):
                    plt.subplot(8,2, j+1)
                    plt.imshow(fakes[j].detach().cpu().numpy().squeeze())
                    plt.yticks([])
                    plt.xticks([])
                    
                plt.subplots_adjust()
                plt.tight_layout()    
                plt.savefig(f'logs_figures/{epoch+1}_{batch_idx}.png')
                plt.close()
                
            step += 1
            gen.train()
           
    loss_gen_list.append(loss_gen_sum/len(train_loader))
    loss_critic_list.append(loss_critic_sum/len(train_loader))
    # scheudulers are optional
    #scheduler1.step()
    #scheduler2.step()

torch.save({
    'gen':gen.state_dict(),
    'critic':critic.state_dict(),
     'opt_gen':opt_gen.state_dict(),
     'opt_critic':opt_critic.state_dict(),
     'epoch':epoch,
     'loss_critc':loss_critic_list,
     'loss_gen':loss_gen_list,
    }, saved_loc)
      
      
               
