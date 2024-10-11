provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

resource "google_compute_network" "vpc_network" {
  name                    = "${var.project_id}-network"
  auto_create_subnetworks = false
  mtu                     = 1460
}

resource "google_compute_subnetwork" "default" {
  name          = "${var.project_id}-subnet"
  ip_cidr_range = var.subnet_cidr
  region        = var.region
  network       = google_compute_network.vpc_network.id
}

resource "google_compute_address" "static_ip" {
  name   = "${var.project_id}-flask-static-ip"
  region = var.region
}

resource "google_compute_instance" "default" {
  name         = "${var.project_id}-flask-vm-p100"
  machine_type = var.machine_type
  zone         = var.zone
  tags         = ["ssh"]

  guest_accelerator {
    type  = "nvidia-tesla-p100"
    count = 1
  }

  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = false
  }

  boot_disk {
    initialize_params {
      image = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"
      size  = var.boot_disk_size
    }
  }

  network_interface {
    subnetwork = google_compute_subnetwork.default.id

    access_config {
      nat_ip = google_compute_address.static_ip.address
    }
  }

  metadata_startup_script = templatefile("${path.module}/startup_script.tftpl", {
    openai_api_key = var.openai_api_key
    repo_url       = var.repo_url
  })

  service_account {
    scopes = [
      "https://www.googleapis.com/auth/compute.readonly",
      "https://www.googleapis.com/auth/logging.write",
      "https://www.googleapis.com/auth/monitoring.write",
      "https://www.googleapis.com/auth/devstorage.read_only"
    ]
  }
}

resource "google_compute_firewall" "flask" {
  name    = "${var.project_id}-allow-flask"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["5050"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

resource "google_compute_firewall" "ssh" {
  name    = "${var.project_id}-allow-ssh"
  network = google_compute_network.vpc_network.name

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["ssh"]
}

output "Web-server-URL" {
  value = "http://${google_compute_address.static_ip.address}:5050"
}