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

2. 