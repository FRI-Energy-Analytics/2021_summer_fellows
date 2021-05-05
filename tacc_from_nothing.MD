# PyTorch on Maverick2
### By Nihal Jere

This is meant to be a introduction to using PyTorch on TACC systems that can be
copied line by line into the shell and work. Throught the text, I will try to
give suggestions for further reading if interested. I will assume that you know
nothing about the shell (better safe than sorry), so if you are familiar with
some basic commands, there are parts that you can skim. Read the [Maverick2
User Guide](https://portal.tacc.utexas.edu/user-guides/maverick2) as a
prerequisite for this. I will assume that you are able to log into Maverick2,
and have gotten your code onto Maverick2. I will also assume basic shell and
Linux knowledge.

Here is the obligatory **DO NOT RUN JOBS ON THE LOGIN NODE**.

### Setting up our environment

Maverick2 doesn't come with PyTorch installation, so we'll have to install it
ourselves. It also doesn't come with pip3, so we need to install it. Since you
will not have permission to install anything globally, we use the `--user` flag
to install it locally:

    $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $ python3 get-pip.py --user
If you now run

    $ ls ~/.local/bin
you should see pip3, wheel and setuptools and maybe a few other programs
installed. By default, `~/.local/bin` is not on the path, so open up your
`.bashrc` in a text editor, and add the following line to the bottom:

    export PATH="$PATH:$HOME/.local/bin"
and reload it using

    $ source ~/.bashrc
Now to install PyTorch, run

    $ pip3 install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html --user
We now have everything that we need to create jobs using PyTorch.

### Starting an interactive session

In order to run PyTorch code on Maverick2, you need your code in a `.py` file.
I will assume the file is named `model.py` for the sake of convenience for the
rest of the tutorial. To insure that nothing we do in this section runs on the
login node, run
    $ idev
while which create an "interactive development" session, essentially reserving
a compute node for half an hour and giving us shell access. It might take a few
seconds, but you'll know when its done when it gives you a shell prompt. To
make sure the python code runs as intended, you can now do
    $ python3 model.py
since you aren't on the login node anymore. If all goes well, we are ready to
create our job. To close your compute node session, just type
    $ exit

### Creating a job

A job file is just a bash script with some specially formatted comments that
give SLURM, the job scheduler, some information about the job.

Here is a sample slurm file `model.slurm`:

    #!/bin/bash
    #----------------------------------------------------
    # Sample Slurm job script
    #----------------------------------------------------

    #SBATCH -J myjob           # Job name
    #SBATCH -o myjob.o%j       # Name of stdout output file
    #SBATCH -e myjob.e%j       # Name of stderr error file
    #SBATCH -p gtx             # Queue (partition) name
    #SBATCH -N 1               # Total # of nodes (must be 1 for serial)
    #SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
    #SBATCH -t 01:30:00        # Run time (hh:mm:ss)
    #SBATCH --mail-user=username@tacc.utexas.edu
    #SBATCH --mail-type=all    # Send email at begin and end of job
    #SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

    module load cuda/10.1
    python3 model.py

Note that the `-A` parameter is optional if you only have one allocation, which
if you're reading this, you probably do. Change the email address to yours, or
get rid of the two email lines completely if you don't want notifications.

Put `model.slurm` and `model.py` in your work directory, as your home directory
should not be used for running jobs. Then, you can add the job to the queue by
using
    $ sbatch model.slurm
To see a list of jobs on the queue, run
    $ squeue
If you see yours, congrats! You have successfully submitted a job!
