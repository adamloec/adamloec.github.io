---
title: UWApp (Universal Web Application)
layout: home
nav_order: 3
parent: Projects
---

[uwapp github]: https://github.com/adamloec/UWApp

# UWApp (Universal Web Application)
{: .no_toc }
{: .fs-9 }
A full-stack web application template for use in future personal projects.
{: .fs-6 .fw-300 }

[Github][uwapp github]{: .btn .fs-5 .mb-4 .mb-md-0 }

## Table of contents
{: .no_toc .text-delta }
1. TOC
{:toc}

---

## About

This documentation contains the README.md contents of my web application template project, UWApp.

---

## Purpose

A personal template repository for future web application development. Easy installation, setup, and configuration for all applications.

---

## Features

- Django backend. Utilizing custom API calls for handling server side information.
- User registration and authentication with login views and models.
- Pay-wall ready custom user model.
- SQLite database.
- Easily add accessory Django applications.
- React frontend with customizable components.

---

## Installation and Initial Setup

- Clone the repository:

```
C:\> git clone https://github.com/adamloec/UWApp
```

- Install the prerequisites:
    - Python 3.7 or later
    - NPM

- Create and activate a virtual environment inside of the repository and install required packages:

```
C:\> cd UWApp
C:\UWApp> virtualenv .venv
C:\UWApp> .venv/Scripts/activate
C:\UWApp> pip install -r requirements.txt
```

- Install and initialize React Packages:

```
C:\UWApp\uwapp\frontend> npm install
C:\UWApp\uwapp\frontend> npm run build
```

- Activate the virtual environment and run the Django backend locally:

```
C:\UWApp> .venv/Scripts/activate
C:\UWApp\uwapp> python manage.py makemigrations uwapp
C:\UWApp\uwapp> python manage.py migrate
C:\UWApp\uwapp> python manage.py runserver
```

- Open a new terminal window and start the frontend react application:

```
C:\UWApp\uwapp\frontend> npm start
```

---

## Submodules

Submodules are used inside of UWApp to manage all peripheral applications. There will be 2 developer cases for managing submodules:

- Working on submodules outside of the main UWApp repository.
- Working on submodules inside of the main UWApp repository.

{: .note}
All submodules will be located inside of *UWApp/uwapp*

### Updating and Managing Submodules

- Merge the latest commits for all submodules:

```
C:\UWApp> git submodule update --remote --merge
```

- Push the latest submodule branch(s) to the main UWApp repository:

```
C:\UWApp> git add --all
C:\UWApp> git commit -m ""
C:\UWApp> git push
```

### Working on Submodules inside of UWApp

After making changes to a submodule located inside of your working UWApp repository, the submodule and main UWApp branch need to be updated for the changes to be tracked.

- Add and push all changes made to the submodule:

```
C:\UWApp\uwapp\submodule> git add {UPDATED_CONTENTS}
C:\UWApp\uwapp\submodule> git commit -m ""
C:\UWApp\uwapp\submodule> git push
```

- After changes were pushed to the submodule repository, fetch the latest commits and push them to the main UWApp repository:

```
C:\UWApp> git fetch
C:\UWApp> git add --all
C:\UWApp> git commit -m ""
C:\UWApp> git push
```