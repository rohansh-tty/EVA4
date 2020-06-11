class CyclicLR:
    def __init__(self, max_lr, min_lr, stepsize, num_iterations):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.stepsize = stepsize
        self.iterations = num_iterations
        self.lr_list = []

    def cycle(self, iteration):
        return int(1 + (iteration/(2*self.stepsize)))

    def lr_position(self, iteration, cycle):
        return abs(iteration/self.stepsize - (2*cycle) + 1)

    def current_lr(self, lr_position):
        return self.min_lr + (self.max_lr - self.min_lr)*(1-lr_position)
    
    def cyclic_lr(self, plotGraph = True):
        for i in range(self.iterations):
            cycle = self.cycle(i)
            lr_position = self.lr_position(i, cycle)
            current_lr = self.current_lr(lr_position)
            self.lr_list.append(current_lr)
        
        if plotGraph:
            fig = plt.figure(figsize=(12,5))
            
            #Plot Title
            plt.title('Cyclic LR Plot')

            plt.xlabel('Iterations')
            plt.ylabel('Learning Rate')

            plt.axhline(y=self.min_lr, label='min lr', color='r')
            plt.text(0, self.min_lr, 'min lr')

            plt.axhline(y=self.max_lr, label='max lr', color='r')
            plt.text(0, self.max_lr, 'max lr')

            plt.plot(self.lr_list)



clr1 = CyclicLR(0.001, 0.0001, len(trainLoader), len(trainLoader)*10)
clr1.cyclic_lr(plotGraph=True)





