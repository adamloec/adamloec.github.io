---
title: CUDA Installation
layout: home
nav_order: 4
has_children: false
---

# CUDA Installation
{: .no_toc }
{: .fs-9 }
Dummy proof guide to install CUDA on anything (with a NVIDIA gpu).
{: .fs-6 .fw-300 }

[Github][grand github]{: .btn .fs-5 .mb-4 .mb-md-0 }

## Table of contents
{: .no_toc .text-delta }
1. TOC
{:toc}

---

## About

I got frustrated with CUDA install guides, so I made this one for my own reference.

---

## Linux Installation

{: .note}
Working as of 11/13/2024. :)

1. Purge/Remove all existing references to CUDA/NVIDIA, if they exist.

{% highlight Bash %}
sudo apt-get purge nvidia*
sudo apt-get autoremove
sudo apt-get autoclean
sudo rm -rf /usr/loca/cuda*
{% endhighlight %}

Reboot machine.

2. Install specific NVIDIA driver version needed for the desired CUDA toolkit version.

{: .note}
Check the CUDA compatibility list and visit the NVIDIA developer repositories page to ensure the version of CUDA is correct for your Ubuntu and Driver version.
CUDA Compatibility List: https://docs.nvidia.com/deploy/cuda-compatibility/
NVIDIA Developer Repositories (apt-key : 3bf863cc.pub):
Ubuntu 22.04:
Ubuntu 24.04:

{% highlight Bash %}
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

sudo apt-get install -y nvidia-driver-{VERSION_NUMBER}
{% endhighlight %}

Reboot machine.

3. Check if driver was installed correctly.

{% highlight Bash %}
nvidia-smi
{% endhighlight %}

4. Fetch the key and repositories from NVIDIA directly.

{% highlight Bash %}
sudo apt-get adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
{% endhighlight %}

5. Install CUDA.

{% highlight Bash %}
sudo apt-get update
sudo apt-get install -y cuda-toolkit-{VERSION}

Example: sudo apt-get install -y cuda-toolkit-12-1
{% endhighlight %}

6. Update the enviornment variables in your .bashrc file.

{% highlight Bash %}
echo 'export PATH=/usr/local/cuda-{VERSION}/bin:$PATH' >> ~/.bashrc 
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-{VERSION}/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
{% endhighlight %}

7. Check if CUDA was installed correctly.

{% highlight Bash %}
nvcc --version
{% endhighlight %}