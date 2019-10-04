# Setting up (pretty sweet) Multi-Hop SSH
For my research, I often have to SSH into the data center at the Colo facility. When I first began, getting access to the online compute server was insanely tedious because I  had to chain these three commands every single time:

```
ssh -Y MYUSERNAME@digilab.astro.berkeley.edu
ssh -Y MYUSERNAME@blph0.ssl.berkeley.edu
ssh -Y blpc0
```
Fortunately, the good folks at Breakthrough Listen (shoutout to Dave Macmahon and Matt Lebofsky) have helped me set this up so I only have to input one command from my local machine.

Now it's my turn to pass on their wisdom to you, gentle reader.

**NOTE**: I worked on a 2017 Macbook Pro at the time I wrote this, so I can't be sure that it's the exact same process in Linux or Windows, but I hope it'll at least get you on the right track.

## Edit  ~/.ssh/config file
First, open up your ~/.ssh/config file. If it's not there, just make a plain text file. You can do this in terminal via `touch ~/.ssh/config` or with whatever text editor you like. Next, you'll want to paste in this code:

```
Host blpc0
  ProxyCommand ssh blph0 -W %h:%p
  User MYUSERNAME
  ForwardX11Trusted yes
Host blph0
  HostName blph0.ssl.berkeley.edu
  User MYUSERNAME
  ProxyCommand ssh digilab -W %h:%p
  ForwardX11Trusted yes
Host digilab
  HostName digilab.astro.berkeley.edu
  User MYUSERNAME
  ForwardX11Trusted yes
```
This allows me to tunnel through `digilab -> blph0 -> blpc0`, which is amazing.
My SSHing abilities aren't too hot, so I'll try to break down what I can.

From my understanding, `HostName` tells the computer the actual address that you're SSHing into. For instance, my original first command was `ssh -Y myusername@digilab.astro.berkeley.edu`, and you can work out that the `HostName` is `digilab.astro.berkeley.edu`.

`ProxyCommand` looks like it's the command prior to the next step of the tunnel. You can see that I don't have a `ProxyCommand` for the bottom SSH, but I do have them for the ones above. Not sure what those flags are, but keep them.

Finally, I think `Host` just sets an alias, and `forwardX11trusted yes` is the same as the `-Y` flag in a regular SSH command.

## SSH-key authentication
Once all that is set up, you'll need to generate SSH public-private key pairs if you want to avoid typing in passwords all at once. The first key-pair comes on your local computer.

If you already have a SSH private key and public key pair that you want to use, you can skip this step. Otherwise, run `ssh-keygen` and you should get this message:

```
Generating public/private rsa key pair.
Enter file in which to save the key (/home/ylo/.ssh/id_rsa):
Enter passphrase (empty for no passphrase): 
Enter same passphrase again: 
Your identification has been saved in id_rsa.
Your public key has been saved in id_rsa.pub.
The key fingerprint is:
SHA256:GKW7yzA1J1qkr1Cr9MhUwAbHbF2NrIPEgZXeOUOz3Us ylo@klar
The key's randomart image is:
+---[RSA 2048]----+
|.*++ o.o.        |
|.+B + oo.        |
| +++ *+.         |
| .o.Oo.+E        |
|    ++B.S.       |
|   o * =.        |
|  + = o          |
| + = = .         |
|  + o o          |
+----[SHA256]-----+
```
Passwords are optional, though I recommend it for your personal security.

Now, you can use the `ssh-copy-id` command like so to copy your RSA key to the server. Since I first SSH to the digilab server, the public key on my machine gets copied there.

The default public key is saved to `id-rsa.pub`, but if you've decided to name it something else, copy with that file instead.

```
ssh-copy-id -i ~/.ssh/id_rsa.pub myusername@digilab.astro.berkeley.edu
```
where you can replace `myusername@digilab.astro.berkeley.edu` with your own username and host.

**IMPORTANT: Do not copy your private key, only your public key (usually ends in .pub)**

You'll know if it worked if you can log in without being asked for a password.
```
ssh -i ~/.ssh/mykey myusername@digilab.astro.berkeley.edu
```

Troubleshooting can be found at https://www.ssh.com/ssh/copy-id, which I totally ripped off.

## Copy keys across machines
By now, hopefully you can SSH into the first machine without having to enter in a password. We're now going to copy that public key originally from the host machine all the way to where you're trying to land.

Check if your key is on the first machine by running
```
cat ~/.ssh/authorized_keys 
```
There should be at least one key there. If not, something's up.

Assuming you've gotten this far, you can just `scp` your public key to as many machines as you want. **Note**: you must ensure that the `.ssh` directory exists in the machines you are copying to—if it doesn't, simply create it!

After verifying that the `.ssh` directory exists on the second machine, run the following command while you're on the first machine that you need to SSH into (usually digilab):

```
scp ~/.ssh/authorized_keys blph0.ssl.berkeley.edu:.ssh/.
```
If you've done this right, you can just use `blph0` or whatever `Host` you've selected in your `~/.ssh/config` file. This command will copy your authorized_keys to the same directory in the host that you set (`blph0` in my case). _Be careful about overwriting!_

Repeat this process of copying your SSH key for however many machines you need to tunnel through.

## Final Examination
If you've gotten through all this successfully, you can test it back on your local machine by using 
```
ssh blpc0
```
or whichever is your final destination. You should be able to log in without having to input any passwords.

I hope this helps you out. I thought it was super rad! Thanks to Dave Macmahon and Matt Lebofsky for helping me set this up.

## Resources
[CERN SSH Automatic Tunneling](https://security.web.cern.ch/security/recommendations/en/ssh_tunneling.shtml)

[How ssh-copy-id works](https://www.ssh.com/ssh/copy-id)

Slack post in #interns page dated February 15, 2018 by Steve Croft

