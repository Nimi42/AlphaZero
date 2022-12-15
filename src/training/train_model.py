from torch import optim, nn
from torch.utils.data import DataLoader


def run_training(model: nn.Module,
        train,
        valid,
        batch_size: int = 32,
        max_epochs: int = 100):

    # Model
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion_pi = nn.CrossEntropyLoss()
    criterion_v = nn.MSELoss()

    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 2,
              'drop_last': True
              }

    training_generator = DataLoader(train, **params)
    validation_generator = DataLoader(valid, **params)

    # Loop over epochs
    for epoch in range(max_epochs):
        running_loss = 0.0

        # Training
        for i, (x, target_pi, target_values) in enumerate(training_generator):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred_pi, pred_v = model(x)

            target_pi = target_pi.view(batch_size, -1)
            loss_pi = criterion_pi(pred_pi, target_pi)
            loss_v = criterion_v(pred_v, target_values)

            total_loss = loss_pi + loss_v
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # print statistics
            running_loss += total_loss.item()
            #if i:# % 2000 == 1999:    # print every 2000 mini-batches
            # print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            # running_loss = 0.0
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')

