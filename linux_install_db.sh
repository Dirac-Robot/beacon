# WARNING: You have to use 'sudo' if you are not in a cluster as a root.
wget -qO- https://download.rethinkdb.com/repository/raw/pubkey.gpg | sudo gpg --dearmor -o /usr/share/keyrings/rethinkdb-archive-keyrings.gpg

# Add the repository.
echo "deb [signed-by=/usr/share/keyrings/rethinkdb-archive-keyrings.gpg] https://download.rethinkdb.com/repository/ubuntu-$(lsb_release -cs) $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rethinkdb.list

sudo apt-get update
sudo apt-get install rethinkdb

# Check installation.
rethinkdb --version
pip3 install rethinkdb
