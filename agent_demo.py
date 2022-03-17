import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        
        self.l1 = nn.Linear(state_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_dim)
        
        # self.l6=nn.BatchNorm1d(state_dim)
        # self.l1 = nn.Linear(state_dim, 256)
        # self.l2=nn.BatchNorm1d(256)
        # self.l3 = nn.Linear(256, 256)
        # self.l4=nn.BatchNorm1d(256)
        # self.l5 = nn.Linear(256, action_dim)

        #self.min_action = torch.FloatTensor(min_action)
        self.max_action = max_action

    def forward(self, state):
        
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        #a = F.relu(self.l3(self.l2(a)))
        return self.max_action * torch.tanh(self.l3(a))
        
        # a = F.relu(self.l2(self.l1(self.l6(state))))
        # a = F.relu(self.l4(self.l3(a)))
        # #a = F.relu(self.l3(self.l2(a)))
        # return self.max_action * torch.tanh(self.l5(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture

        self.l1 = nn.Linear(state_dim + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 1)
        
        # self.l1 = nn.Linear(state_dim + action_dim, 256)
        # self.l2=nn.BatchNorm1d(256)
        # self.l3 = nn.Linear(256, 256)
        # self.l4=nn.BatchNorm1d(256)
        # self.l5 = nn.Linear(256, 1)

        # Q2 architecture
        # self.l4 = nn.Linear(state_dim + action_dim, 8)
        # self.l5 = nn.Linear(8, 8)
        # self.l6 = nn.Linear(8, 1)
        
        # self.l6 = nn.Linear(state_dim + action_dim, 256)
        # self.l7=nn.BatchNorm1d(256)
        # self.l8 = nn.Linear(256, 256)
        # self.l9=nn.BatchNorm1d(256)
        # self.l10 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        # q2 = F.relu(self.l4(sa))
        # q2 = F.relu(self.l5(q2))
        # q2 = self.l6(q2)
        
        # q1 = F.relu(self.l2(self.l1(sa)))
        # q1 = F.relu(self.l4(self.l3(q1)))
        # q1 = self.l5(q1)

        # q2 = F.relu(self.l7(self.l6(sa)))
        # q2 = F.relu(self.l9(self.l8(q2)))
        # q2 = self.l10(q2)
        
        return q1

    # def Q1(self, state, action):
    #     sa = torch.cat([state, action], 1)

    #     q1 = F.relu(self.l1(sa))
    #     q1 = F.relu(self.l2(q1))
    #     q1 = self.l3(q1)
        
    #     # q1 = F.relu(self.l2(self.l1(sa)))
    #     # q1 = F.relu(self.l4(self.l3(q1)))
    #     # q1 = self.l5(q1)
    #     return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.01,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
    ):

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        #self.min_action = torch.FloatTensor(min_action)
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.agent_percent=0.75
        self.demo_percent=0.25

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        # state = state.float()
        
        #Changes to turn off batch-normalization during inference
        #self.actor.eval()
        #action=-self.actor(state).cpu().data.numpy().flatten()
        #self.actor.train()
        #return action
        action = self.actor(state).cpu().data.numpy().flatten()
        #print("Predict: ", state, " ->  ",action)
        target_action = self.actor_target(state).cpu().data.numpy().flatten()
        #print("Target : ", state, " ->  ",target_action)
        return action

    def train(self, replay_buffer,replay_buffer_demo, pretrain=False,batch_size=64):
        self.total_it += 1

        # Sample replay buffer
        if pretrain==True:
           state, action, next_state, reward, not_done = replay_buffer_demo.sample(batch_size) 
        else:   
           agent_batch_size= (int)(self.agent_percent*batch_size)
           demo_batch_size=(int)(self.demo_percent*batch_size)
           state, action, next_state, reward, not_done = replay_buffer.sample(agent_batch_size)
           state_d, action_d, next_state_d, reward_d, not_done_d = replay_buffer_demo.sample(demo_batch_size)
           #Concatenate the agent and human demonstrations; needs to be activated when Q-filter is not required
           state=torch.cat((state,state_d),0)
           action=torch.cat((action,action_d),0)
           next_state=torch.cat((next_state,next_state_d),0)
           reward=torch.cat((reward,reward_d),0)
           not_done=torch.cat((not_done,not_done_d),0)
           #print ("The dimensions of the state are",state.size(),reward.size(),next_state.size())
           
        next_action = self.actor_target(next_state)
        
        #Calculate the behavior cloning (BC) loss from the demonstrations
        if pretrain==False:
           target_action_d=self.actor_target(state_d)
           bc_loss=F.mse_loss(action_d, target_action_d)
        
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            # noise = (torch.randn_like(action) * self.policy_noise).clamp(
            #     -self.noise_clip, self.noise_clip
            # )

            #next_action = (self.actor_target(next_state) + noise).clamp(
            #    -self.max_action, self.max_action
            #)
            
            # target = self.actor_target(next_state)
            # print(noise, " -> ", target,  "  ==  ", noise+target)
             # + noise).clamp(self.min_action[0], self.max_action[0])

            # Compute the target Q value

            next_target_Q = self.critic_target(next_state, next_action)
        target_Q = reward + not_done * self.discount * next_target_Q

        #print("Fut: ",next_state[0], " ->  ",next_action[0], "  ==  ", next_target_Q[0])

        # Get current Q estimates
        current_Q1 = self.critic(state, action)

        # print(state, action, current_Q1, target_Q)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q)
        #print("critic_loss: ", critic_loss)
        #self.critic_loss.append(critic_loss.detach().numpy())

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic.parameters(), 1)
        self.critic_optimizer.step()
        # Delayed policy updates
        # if self.total_it % self.policy_freq == 0:
        # Compute actor loss
        if pretrain==True:
           actor_loss = -self.critic(state, self.actor(state)).mean()
        else:
           #print ("Training the behavior cloning loss") 
           actor_loss = -self.critic(state, self.actor(state)).mean() + bc_loss
        #Append the actor and the critic loss
        #self.actor_loss.append(actor_loss.detach().numpy())
        #self.critic_loss.append(critic_loss.detach().numpy())
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        #print("actor_loss: ", actor_loss)

        # Update the frozen target models
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        print (filename)
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer")
        )
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
        
    def load_demonstration(self, filename):
        print (filename)
        self.demo=np.load(filename,allow_pickle=True)    
        
        
